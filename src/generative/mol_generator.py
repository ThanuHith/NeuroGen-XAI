"""Phase 4A: Constrained SELFIES/RL-Guided De Novo CNS Molecule Generator.

Optimizes novel CNS drug candidates around Blood-Brain Barrier permeability,
Human AChE potency (pIC50), QED drug-likeness, and Synthetic Accessibility (SA),
while enforcing hard medicinal chemistry filters (PAINS, Brenk, hERG risk, SA_Score)
and verifying structural novelty against Phase 1 training sets via Morgan Tanimoto.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import selfies as sf
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, FilterCatalog
from rdkit.Chem.Draw import rdMolDraw2D

from src.data_pipeline.prepare_datasets import FeatureExtractor, MolCleaner
from src.xai.explainer import DONEPEZIL_SMILES, NeuroGenExplainer, save_dual_xai_figure


LOGGER = logging.getLogger("neurogen_xai.generative")

DEFAULT_CNS_SEEDS: dict[str, str] = {
    "Donepezil": DONEPEZIL_SMILES,
    "Tacrine": "Nc1c2ccccc2nc2c1CCCC2",
    "Rivastigmine": "CCN(C)C(=O)Oc1cccc([C@H](C)N(C)C)c1",
    "Galantamine": "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
}


@dataclass
class GenerationConstraints:
    """Multi-objective optimization weights and medicinal chemistry constraints."""

    w_pic50: float = 0.40
    w_bbb: float = 0.35
    w_qed: float = 0.15
    w_sa: float = 0.10
    max_sa_score: float = 4.5
    min_mw: float = 150.0
    max_mw: float = 550.0
    min_heavy_atoms: int = 12
    max_heavy_atoms: int = 42
    max_tanimoto_novelty_cutoff: float = 0.999
    enforce_pains_filter: bool = True
    enforce_brenk_filter: bool = True
    enforce_herg_filter: bool = True
    herg_logp_threshold: float = 4.0
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = {
                "pic50": self.w_pic50,
                "bbb": self.w_bbb,
                "qed": self.w_qed,
                "sa": self.w_sa,
            }


class StructuralAlertFilter:
    """Filters molecules for PAINS, Brenk toxicophores, and putative hERG alerts."""

    def __init__(self, herg_logp_threshold: float = 4.0) -> None:
        self.herg_logp_threshold = herg_logp_threshold

        pains_params = FilterCatalog.FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains_catalog = FilterCatalog.FilterCatalog(pains_params)

        brenk_params = FilterCatalog.FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        self.brenk_catalog = FilterCatalog.FilterCatalog(brenk_params)

        # Basic aliphatic amine (excluding amides/aromatic nitrogens) and 2-4 carbon aromatic tethers
        self._basic_amine = Chem.MolFromSmarts("[#7;+0,+1;!$([N]C=O);!$([N]S=O);!$([n])]")
        self._herg_tethers = [
            Chem.MolFromSmarts("[#7;+0,+1;!$([N]C=O);!$([n])]~[#6]~[#6]~[a]"),
            Chem.MolFromSmarts("[#7;+0,+1;!$([N]C=O);!$([n])]~[#6]~[#6]~[#6]~[a]"),
            Chem.MolFromSmarts("[#7;+0,+1;!$([N]C=O);!$([n])]~[#6]~[#6]~[#6]~[#6]~[a]"),
        ]

    def passes_pains(self, mol: Chem.Mol) -> bool:
        """Return True if molecule has zero PAINS alerts."""
        return not self.pains_catalog.HasMatch(mol)

    def passes_brenk(self, mol: Chem.Mol) -> bool:
        """Return True if molecule has zero Brenk medicinal chemistry alerts."""
        return not self.brenk_catalog.HasMatch(mol)

    def passes_herg_heuristic(self, mol: Chem.Mol, logp: float) -> bool:
        """Return True if molecule avoids elevated lipophilic basic-amine hERG risk."""
        has_amine = self._basic_amine is not None and mol.HasSubstructMatch(self._basic_amine)
        if not has_amine:
            return True
        if logp > 5.0:
            return False
        if logp > self.herg_logp_threshold:
            for pattern in self._herg_tethers:
                if pattern is not None and mol.HasSubstructMatch(pattern):
                    return False
        return True

    def inspect(
        self,
        mol: Chem.Mol,
        logp: float,
        sa_score: float,
        max_sa_score: float = 4.5,
        enforce_pains: bool = True,
        enforce_brenk: bool = True,
        enforce_herg: bool = True,
    ) -> dict[str, bool]:
        """Run all medicinal chemistry and synthesizability checks."""
        pains_ok = self.passes_pains(mol)
        brenk_ok = self.passes_brenk(mol)
        herg_ok = self.passes_herg_heuristic(mol, logp=logp)
        sa_ok = sa_score <= max_sa_score

        passes_all = (
            (pains_ok or not enforce_pains)
            and (brenk_ok or not enforce_brenk)
            and (herg_ok or not enforce_herg)
            and sa_ok
        )
        return {
            "passes_pains": pains_ok,
            "passes_brenk": brenk_ok,
            "passes_herg": herg_ok,
            "passes_sa": sa_ok,
            "passes_all": passes_all,
        }


def compute_multi_objective_reward(
    pic50: float,
    bbb_prob: float,
    qed: float,
    sa_score: float,
    constraints: GenerationConstraints,
) -> float:
    """Compute multi-objective CNS optimization reward."""
    norm_pic50 = float(np.clip((pic50 - 4.0) / 6.0, 0.0, 1.0))
    norm_sa = float(np.clip((sa_score - 1.0) / 9.0, 0.0, 1.0))
    return float(
        constraints.w_pic50 * norm_pic50
        + constraints.w_bbb * bbb_prob
        + constraints.w_qed * qed
        - constraints.w_sa * norm_sa
    )


class ConstrainedSELFIESGenerator:
    """Evolutionary / RL-guided SELFIES molecular generator with ensemble scoring."""

    def __init__(
        self,
        explainer: NeuroGenExplainer | None = None,
        constraints: GenerationConstraints | None = None,
        processed_dir: Path | str = Path("data/processed"),
        seed: int = 42,
    ) -> None:
        self.explainer = explainer or NeuroGenExplainer()
        self.constraints = constraints or GenerationConstraints()
        self.processed_dir = Path(processed_dir)
        self.rng = np.random.default_rng(seed)

        self.cleaner = MolCleaner()
        self.extractor = FeatureExtractor()
        self.alert_filter = StructuralAlertFilter(
            herg_logp_threshold=self.constraints.herg_logp_threshold
        )

        self.train_smiles_set, self.train_fp_matrix = self._load_reference_corpus()
        self.train_fp_sum = self.train_fp_matrix.sum(axis=1, keepdims=True).T
        self.alphabet = self._build_cns_selfies_alphabet()

    def _load_reference_corpus(self) -> tuple[set[str], np.ndarray]:
        """Load canonical SMILES & 2048-bit Morgan FPs from Phase 1 datasets."""
        smiles_set: set[str] = set()
        fp_blocks: list[np.ndarray] = []

        for filename in ("bbbp_features.npz", "chembl220_features.npz"):
            path = self.processed_dir / filename
            if not path.exists():
                LOGGER.warning("Reference NPZ not found for novelty check: %s", path)
                continue
            with np.load(path, allow_pickle=False) as archive:
                smiles_set.update(str(s) for s in archive["smiles"])
                fp_blocks.append(archive["X_fp"].astype(np.float32))

        if not fp_blocks:
            return smiles_set, np.zeros((0, 2048), dtype=np.float32)

        fp_matrix = np.vstack(fp_blocks).astype(np.float32)
        LOGGER.info(
            "Loaded reference training corpus for novelty check: %d unique SMILES (%d FPs)",
            len(smiles_set),
            len(fp_matrix),
        )
        return smiles_set, fp_matrix

    @staticmethod
    def _build_cns_selfies_alphabet() -> list[str]:
        """Build a clean organic CNS-focused SELFIES token vocabulary."""
        allowed_elements = {"C", "N", "O", "F", "S", "Cl", "Br"}
        tokens: set[str] = set()

        for smi in DEFAULT_CNS_SEEDS.values():
            try:
                enc = sf.encoder(smi)
                if enc:
                    tokens.update(sf.split_selfies(enc))
            except Exception:
                continue

        # Curated medicinal chemistry organic & topology tokens
        curated_tokens = [
            "[C]", "[=C]", "[#C]", "[N]", "[=N]", "[#N]", "[O]", "[=O]",
            "[F]", "[Cl]", "[S]", "[=S]",
            "[Branch1]", "[Branch2]", "[Ring1]", "[Ring2]",
            "[C][=C]", "[C][=O]", "[O][C]", "[C][F]",
        ]
        for tok in curated_tokens:
            if tok.count("[") == 1:
                tokens.add(tok)

        for tok in sf.get_semantic_robust_alphabet():
            if tok.startswith("[Branch") or tok.startswith("[Ring"):
                tokens.add(tok)
                continue
            stripped = tok.strip("[]=#+-0123456789H")
            if stripped in allowed_elements and "+" not in tok and "-" not in tok:
                tokens.add(tok)

        return sorted(tokens)

    def _compute_max_tanimoto_batch(self, fp_batch: np.ndarray) -> np.ndarray:
        """Compute exact max Morgan Tanimoto similarity vs. all training molecules."""
        if self.train_fp_matrix.shape[0] == 0 or fp_batch.shape[0] == 0:
            return np.zeros(fp_batch.shape[0], dtype=np.float64)

        cand = fp_batch.astype(np.float32)
        inter = cand @ self.train_fp_matrix.T
        cand_sum = cand.sum(axis=1, keepdims=True)
        union = np.maximum(cand_sum + self.train_fp_sum - inter, 1.0)
        sim_matrix = inter / union
        return np.max(sim_matrix, axis=1).astype(np.float64)

    def _mutate_selfies(
        self,
        tokens: list[str],
        token_weights: np.ndarray,
        n_edits: int = 2,
    ) -> list[str]:
        """Apply policy-weighted SELFIES token substitution, insertion, or deletion."""
        out = list(tokens)
        probs = token_weights / np.sum(token_weights)

        for _ in range(n_edits):
            if not out:
                out.append(self.alphabet[int(self.rng.choice(len(self.alphabet), p=probs))])
                continue
            op = self.rng.choice(["replace", "insert", "delete"], p=[0.50, 0.30, 0.20])
            pos = int(self.rng.integers(0, len(out)))
            if op == "replace":
                out[pos] = self.alphabet[int(self.rng.choice(len(self.alphabet), p=probs))]
            elif op == "insert" and len(out) < 75:
                new_tok = self.alphabet[int(self.rng.choice(len(self.alphabet), p=probs))]
                out.insert(pos, new_tok)
            elif op == "delete" and len(out) > 8:
                out.pop(pos)
        return out

    def _crossover_selfies(self, tokens_a: list[str], tokens_b: list[str]) -> list[str]:
        """Perform 1-point crossover between two parent SELFIES token sequences."""
        if len(tokens_a) < 4 or len(tokens_b) < 4:
            return list(tokens_a)
        cut_a = int(self.rng.integers(2, len(tokens_a) - 1))
        cut_b = int(self.rng.integers(2, len(tokens_b) - 1))
        child = tokens_a[:cut_a] + tokens_b[cut_b:]
        return child[:80]

    def _canonicalize_and_prefilter(
        self,
        raw_smiles: str,
    ) -> tuple[str, Chem.Mol, dict[str, float], np.ndarray, dict[str, bool]] | None:
        """Standardize decoded SMILES and enforce structural/synthesizability filters."""
        canon, mol, err = self.cleaner.standardize_smiles(raw_smiles)
        if err is not None or canon is None or mol is None:
            return None

        if canon in self.train_smiles_set:
            return None

        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < self.constraints.min_heavy_atoms or n_heavy > self.constraints.max_heavy_atoms:
            return None

        if mol.GetRingInfo().NumRings() < 1:
            return None

        desc = self.extractor.compute_descriptors(mol)
        if desc["MW"] < self.constraints.min_mw or desc["MW"] > self.constraints.max_mw:
            return None

        alert_status = self.alert_filter.inspect(
            mol,
            logp=desc["LogP"],
            sa_score=desc["SA_Score"],
            max_sa_score=self.constraints.max_sa_score,
            enforce_pains=self.constraints.enforce_pains_filter,
            enforce_brenk=self.constraints.enforce_brenk_filter,
            enforce_herg=self.constraints.enforce_herg_filter,
        )
        if not alert_status["passes_all"]:
            return None

        fp = self.extractor.compute_morgan_fp(mol).astype(np.float32)
        return canon, mol, desc, fp, alert_status

    def generate(
        self,
        seed_smiles: list[str] | dict[str, str] | None = None,
        generations: int = 16,
        population_size: int = 48,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Run RL/evolutionary SELFIES optimization and return Top-K novel candidates."""
        if seed_smiles is None:
            seed_map = dict(DEFAULT_CNS_SEEDS)
        elif isinstance(seed_smiles, dict):
            seed_map = dict(seed_smiles)
        else:
            seed_map = {f"Seed_{i + 1}": s for i, s in enumerate(seed_smiles)}

        # Encode valid seed scaffolds to SELFIES
        parent_pool: list[tuple[list[str], float, str]] = []
        for seed_name, smi in seed_map.items():
            canon, _, err = self.cleaner.standardize_smiles(smi)
            if err is not None or canon is None:
                continue
            try:
                enc = sf.encoder(canon)
                if enc:
                    toks = list(sf.split_selfies(enc))
                    pred = self.explainer.predict_molecule(canon)
                    d = pred["descriptors"]
                    reward = compute_multi_objective_reward(
                        pred["pic50_ensemble"],
                        pred["bbb_prob_ensemble"],
                        d["QED"],
                        d["SA_Score"],
                        self.constraints,
                    )
                    parent_pool.append((toks, reward, seed_name))
            except Exception:
                continue

        if not parent_pool:
            raise RuntimeError("No valid seed SMILES could be encoded to SELFIES.")

        # Policy weights over SELFIES alphabet updated via REINFORCE-style reward feedback
        token_to_idx = {tok: i for i, tok in enumerate(self.alphabet)}
        token_weights = np.ones(len(self.alphabet), dtype=np.float64)

        discovered: dict[str, dict[str, Any]] = {}
        seen_smiles: set[str] = set(self.train_smiles_set)

        for gen in range(1, generations + 1):
            # Softmax sampling probabilities over current parent pool
            rewards_arr = np.array([p[1] for p in parent_pool], dtype=np.float64)
            logits = (rewards_arr - np.max(rewards_arr)) * 8.0
            parent_probs = np.exp(logits)
            parent_probs /= np.sum(parent_probs)

            candidate_items: list[tuple[str, list[str], str, dict[str, float], np.ndarray, dict[str, bool]]] = []
            attempts = 0
            max_attempts = population_size * 8

            while len(candidate_items) < population_size and attempts < max_attempts:
                attempts += 1
                idx_a = int(self.rng.choice(len(parent_pool), p=parent_probs))
                toks_a, _, origin_a = parent_pool[idx_a]

                if len(parent_pool) >= 2 and self.rng.random() < 0.35:
                    idx_b = int(self.rng.choice(len(parent_pool), p=parent_probs))
                    toks_b, _, _ = parent_pool[idx_b]
                    child_toks = self._crossover_selfies(toks_a, toks_b)
                    n_edits = int(self.rng.integers(1, 3))
                else:
                    child_toks = toks_a
                    n_edits = int(self.rng.integers(1, 4))

                mutated_toks = self._mutate_selfies(child_toks, token_weights, n_edits=n_edits)
                selfies_str = "".join(mutated_toks)
                try:
                    decoded_smi = sf.decoder(selfies_str)
                except Exception:
                    continue
                if not decoded_smi:
                    continue

                prefiltered = self._canonicalize_and_prefilter(decoded_smi)
                if prefiltered is None:
                    continue

                canon, _, desc, fp, alert_status = prefiltered
                if canon in seen_smiles:
                    continue
                seen_smiles.add(canon)
                candidate_items.append((canon, mutated_toks, origin_a, desc, fp, alert_status))

            if not candidate_items:
                LOGGER.info("Generation %02d/%02d: 0 novel filtered candidates.", gen, generations)
                continue

            # Check Morgan Tanimoto novelty against Phase 1 training corpus
            fp_batch = np.vstack([item[4] for item in candidate_items])
            max_tanimotos = self._compute_max_tanimoto_batch(fp_batch)

            novel_indices = [
                i
                for i, sim in enumerate(max_tanimotos)
                if sim < self.constraints.max_tanimoto_novelty_cutoff
            ]
            if not novel_indices:
                continue

            novel_smiles = [candidate_items[i][0] for i in novel_indices]
            preds = self.explainer.predict_batch(novel_smiles)

            gen_rewards: list[float] = []
            for idx_in_novel, orig_i in enumerate(novel_indices):
                pred = preds[idx_in_novel]
                if not pred.get("valid", False):
                    continue

                canon, toks, origin, desc, _, alert_status = candidate_items[orig_i]
                sim = float(max_tanimotos[orig_i])
                reward = compute_multi_objective_reward(
                    pred["pic50_ensemble"],
                    pred["bbb_prob_ensemble"],
                    desc["QED"],
                    desc["SA_Score"],
                    self.constraints,
                )
                gen_rewards.append(reward)

                record = {
                    "canonical_smiles": canon,
                    "selfies": "".join(toks),
                    "reward": round(reward, 4),
                    "pic50_ensemble": round(pred["pic50_ensemble"], 4),
                    "ic50_nm": round(pred["ic50_nm"], 2),
                    "bbb_prob_ensemble": round(pred["bbb_prob_ensemble"], 4),
                    "bbb_prob_rf": round(pred["bbb_prob_rf"], 4),
                    "bbb_prob_xgb": round(pred["bbb_prob_xgb"], 4),
                    "bbb_prob_gnn": round(pred["bbb_prob_gnn"], 4),
                    "pic50_rf": round(pred["pic50_rf"], 4),
                    "pic50_xgb": round(pred["pic50_xgb"], 4),
                    "pic50_gnn": round(pred["pic50_gnn"], 4),
                    "qed": round(desc["QED"], 4),
                    "sa_score": round(desc["SA_Score"], 4),
                    "mw": round(desc["MW"], 2),
                    "logp": round(desc["LogP"], 3),
                    "tpsa": round(desc["TPSA"], 2),
                    "hbd": int(desc["HBD"]),
                    "hba": int(desc["HBA"]),
                    "rot_bonds": int(desc["RotBonds"]),
                    "lipinski_pass": bool(pred["lipinski_pass"]),
                    "passes_pains": bool(alert_status["passes_pains"]),
                    "passes_brenk": bool(alert_status["passes_brenk"]),
                    "passes_herg": bool(alert_status["passes_herg"]),
                    "max_train_tanimoto": round(sim, 4),
                    "novelty_score": round(1.0 - sim, 4),
                    "seed_origin": origin,
                    "generation": gen,
                }
                discovered[canon] = record
                parent_pool.append((toks, reward, origin))

                # Policy gradient-style reinforcement on tokens in high-reward molecules
                if reward >= float(np.median(rewards_arr)):
                    boost = 0.25 * max(reward, 0.0)
                    for tok in set(toks):
                        if tok in token_to_idx:
                            token_weights[token_to_idx[tok]] += boost

            # Retain top-32 elite parents across generations
            parent_pool.sort(key=lambda item: item[1], reverse=True)
            parent_pool = parent_pool[:32]
            if gen_rewards:
                LOGGER.info(
                    "Gen %02d/%02d | Valid Novel=%2d | Best Gen Reward=%.4f | Global Best=%.4f",
                    gen,
                    generations,
                    len(gen_rewards),
                    max(gen_rewards),
                    max(r["reward"] for r in discovered.values()),
                )

        if not discovered:
            raise RuntimeError("De novo generator produced 0 valid filtered candidates.")

        df = pd.DataFrame(discovered.values()).sort_values("reward", ascending=False).head(top_k)
        df = df.reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df


def render_candidates_grid(
    candidates_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Render a 300 DPI 2x5 visual grid of the top generated CNS molecules."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_mols = min(len(candidates_df), 10)
    n_cols = 5 if n_mols > 5 else max(n_mols, 1)
    n_rows = int(np.ceil(n_mols / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 4.0 * n_rows), dpi=300)
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes_flat):
        if idx >= n_mols:
            ax.axis("off")
            continue

        row = candidates_df.iloc[idx]
        mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        if mol is None:
            ax.axis("off")
            continue

        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(420, 320)
        drawer.drawOptions().clearBackground = True
        drawer.drawOptions().padding = 0.08
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(BytesIO(drawer.GetDrawingText()))

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#d0d7de")
            spine.set_linewidth(1.0)

        ax.set_title(
            f"#{int(row['rank'])} | Reward: {row['reward']:.3f} ({row['seed_origin']})",
            fontsize=9.5,
            fontweight="bold",
            pad=6,
        )
        caption = (
            f"pIC50: {row['pic50_ensemble']:.2f} ({row['ic50_nm']:.0f} nM) | "
            f"P(BBB+): {row['bbb_prob_ensemble']:.2f}\n"
            f"QED: {row['qed']:.2f} | SA: {row['sa_score']:.2f} | "
            f"MaxSim: {row['max_train_tanimoto']:.2f}"
        )
        ax.set_xlabel(caption, fontsize=8.0, labelpad=6)

    fig.suptitle(
        "NeuroGen-XAI Top Constrained De Novo CNS Candidates (PAINS/Brenk/hERG/SA Filtered)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved generated molecules grid -> %s", out_path)
    return out_path


def run_cli(
    generations: int = 16,
    population_size: int = 48,
    top_k: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Phase 3 Donepezil XAI verification and Phase 4 constrained generation."""
    explainer = NeuroGenExplainer()

    # Phase 3 verification artifact: Donepezil dual-XAI attribution figure
    donepezil_attr = explainer.explain_molecule(DONEPEZIL_SMILES)
    save_dual_xai_figure(
        donepezil_attr,
        output_path=Path("reports/figures/xai_donepezil_attribution.png"),
        molecule_title="Donepezil",
    )

    # Phase 4 constrained SELFIES generation
    generator = ConstrainedSELFIESGenerator(explainer=explainer, seed=seed)
    top_df = generator.generate(
        seed_smiles=DEFAULT_CNS_SEEDS,
        generations=generations,
        population_size=population_size,
        top_k=top_k,
    )

    csv_path = Path("reports/generated_cns_candidates.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(csv_path, index=False)
    LOGGER.info("Exported Top %d novel CNS candidates -> %s", len(top_df), csv_path)

    render_candidates_grid(top_df, Path("reports/figures/top_generated_molecules.png"))
    return top_df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for constrained de novo generation."""
    parser = argparse.ArgumentParser(
        description="NeuroGen-XAI Phase 4: Constrained SELFIES De Novo CNS Generator"
    )
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--population-size", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    df_top = run_cli(
        generations=args.generations,
        population_size=args.population_size,
        top_k=args.top_k,
        seed=args.seed,
    )
    print("\nTop 10 Generated Novel CNS Candidates:")
    cols = [
        "rank",
        "reward",
        "pic50_ensemble",
        "bbb_prob_ensemble",
        "qed",
        "sa_score",
        "max_train_tanimoto",
        "seed_origin",
        "canonical_smiles",
    ]
    print(df_top[cols].to_string(index=False))
