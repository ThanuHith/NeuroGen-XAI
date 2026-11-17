"""Phase 3: Explainable AI (XAI) & Ensemble Inference Module for NeuroGen-XAI.

Provides:
  1. Unified ensemble inference across all 6 Phase 2 checkpoints:
       - BBBP Classification: rf_bbbp.joblib, xgb_bbbp.joblib, best_gnn_bbbp.pt
       - CHEMBL220 pIC50 Regression: rf_ache.joblib, xgb_ache.joblib, best_gnn_ache.pt
  2. Dual XAI Atom-Level Attribution:
       - TreeSHAP (XGBoost exact pred_contribs / shap.TreeExplainer) + Random Forest
         active-bit perturbation mapped back to atoms via RDKit Morgan `bitInfo`.
       - GNN Input*Gradient saliency + batched atom-occlusion delta from GINEConv.
       - Fused normalized per-atom importance weights in [-1, +1].
  3. High-resolution 2D RDKit atom-attribution heatmaps (PNG/SVG) highlighting
     pro-potency/BBB+ substructures in green and limiting substructures in red.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch_geometric.loader import DataLoader
from xgboost import XGBClassifier, XGBRegressor

from src.data_pipeline.prepare_datasets import (
    DESCRIPTOR_COLUMNS,
    MORGAN_NBITS,
    MORGAN_RADIUS,
    FeatureExtractor,
    MolCleaner,
)
from src.models.baseline_rf_xgb import load_scaffold_split_data
from src.models.gnn_multitask import GINEConfig, NeuroGINEModel, smiles_to_pyg_data

try:
    import shap  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional if no wheel on platform
    shap = None


LOGGER = logging.getLogger("neurogen_xai.xai")

DONEPEZIL_SMILES = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"


@dataclass
class AttributionResult:
    """Container for ensemble predictions, SHAP descriptor importances, and atom weights."""

    canonical_smiles: str
    prediction: dict[str, Any]
    atom_weights_bbb: list[float]
    atom_weights_ache: list[float]
    shap_atom_weights_bbb: list[float]
    gnn_atom_weights_bbb: list[float]
    shap_atom_weights_ache: list[float]
    gnn_atom_weights_ache: list[float]
    descriptor_shap_bbb: dict[str, float]
    descriptor_shap_ache: dict[str, float]
    png_bbb: bytes
    png_ache: bytes
    svg_bbb: str
    svg_ache: str


def _normalize_signed(vec: np.ndarray) -> np.ndarray:
    """Scale a vector by its max absolute value into [-1.0, +1.0]."""
    arr = np.asarray(vec, dtype=np.float64)
    if arr.size == 0:
        return arr
    max_abs = float(np.max(np.abs(arr)))
    if max_abs < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip(arr / max_abs, -1.0, 1.0)


def _compute_lipinski_status(desc: dict[str, float]) -> tuple[int, bool]:
    """Evaluate Lipinski Rule-of-5 violations (pass if <= 1 violation)."""
    violations = 0
    if desc["MW"] > 500.0:
        violations += 1
    if desc["LogP"] > 5.0:
        violations += 1
    if desc["HBD"] > 5.0:
        violations += 1
    if desc["HBA"] > 10.0:
        violations += 1
    return violations, violations <= 1


class NeuroGenExplainer:
    """Loads Phase 2 ensemble checkpoints and computes predictions & dual-XAI maps."""

    def __init__(
        self,
        checkpoint_dir: Path | str = Path("models/checkpoints"),
        data_dir: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.data_dir = (
            Path(data_dir)
            if data_dir is not None
            else self.checkpoint_dir.resolve().parent.parent / "data" / "processed"
        )
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cleaner = MolCleaner()
        self.extractor = FeatureExtractor(radius=MORGAN_RADIUS, n_bits=MORGAN_NBITS)

        # Load or self-heal baseline joblib bundles (handles cloud cold-starts & sklearn version drift)
        rf_bbbp_bundle, xgb_bbbp_bundle = self._load_or_fit_task_baselines("bbbp")
        rf_ache_bundle, xgb_ache_bundle = self._load_or_fit_task_baselines("chembl220")

        self.rf_bbbp = rf_bbbp_bundle["model"]
        self.xgb_bbbp = xgb_bbbp_bundle["model"]
        self.bbbp_scaler = rf_bbbp_bundle["scaler"]

        self.rf_ache = rf_ache_bundle["model"]
        self.xgb_ache = xgb_ache_bundle["model"]
        self.ache_scaler = rf_ache_bundle["scaler"]

        # Load PyTorch Geometric GINEConv models (with graceful fallback if missing)
        self.gnn_bbbp, self.gnn_bbbp_mean, self.gnn_bbbp_scale = self._load_gnn("best_gnn_bbbp.pt")
        self.gnn_ache, self.gnn_ache_mean, self.gnn_ache_scale = self._load_gnn("best_gnn_ache.pt")

        # Initialize TreeExplainer when `shap` package is available
        self._shap_xgb_bbbp = self._init_tree_explainer(self.xgb_bbbp)
        self._shap_xgb_ache = self._init_tree_explainer(self.xgb_ache)

    @staticmethod
    def _init_tree_explainer(model: Any) -> Any | None:
        if shap is None:
            return None
        try:
            return shap.TreeExplainer(model)
        except Exception as exc:
            LOGGER.warning("Falling back to XGBoost pred_contribs TreeSHAP: %s", exc)
            return None

    def _try_load_joblib(self, filename: str) -> dict[str, Any] | None:
        path = self.checkpoint_dir / filename
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception as exc:
            LOGGER.warning("Could not unpickle %s (%s); will refit from NPZ.", path, exc)
            return None

    def _load_or_fit_task_baselines(self, task: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load RF/XGB checkpoints or refit quickly from Phase 1 NPZ archives if absent/incompatible."""
        if task == "bbbp":
            rf_file, xgb_file, npz_file = "rf_bbbp.joblib", "xgb_bbbp.joblib", "bbbp_features.npz"
        else:
            rf_file, xgb_file, npz_file = "rf_ache.joblib", "xgb_ache.joblib", "chembl220_features.npz"

        rf_bundle = self._try_load_joblib(rf_file)
        xgb_bundle = self._try_load_joblib(xgb_file)
        if rf_bundle is not None and xgb_bundle is not None:
            return rf_bundle, xgb_bundle

        npz_path = self.data_dir / npz_file
        if not npz_path.exists():
            npz_path = Path("data/processed") / npz_file
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint(s) for {task} and fallback feature archive not found at {npz_path}"
            )

        ds = load_scaffold_split_data(npz_path, dataset_name=task)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if rf_bundle is None:
            LOGGER.info("Fitting fast fallback RandomForest (%s) from %s...", task, npz_path)
            if task == "bbbp":
                rf_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                rf_model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )
            rf_model.fit(ds.X_train, ds.y_train)
            rf_bundle = {
                "model": rf_model,
                "scaler": ds.scaler,
                "descriptor_names": ds.descriptor_names,
            }
            try:
                joblib.dump(rf_bundle, self.checkpoint_dir / rf_file, compress=3)
            except Exception as exc:
                LOGGER.warning("Could not cache %s: %s", rf_file, exc)

        if xgb_bundle is None:
            LOGGER.info("Fitting fast fallback XGBoost (%s) from %s...", task, npz_path)
            if task == "bbbp":
                n_pos = int((ds.y_train == 1).sum())
                n_neg = int((ds.y_train == 0).sum())
                xgb_model = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=float(n_neg / max(n_pos, 1)),
                    eval_metric="aucpr",
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                xgb_model = XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="rmse",
                    random_state=42,
                    n_jobs=-1,
                )
            xgb_model.fit(ds.X_train, ds.y_train)
            xgb_bundle = {
                "model": xgb_model,
                "scaler": ds.scaler,
                "descriptor_names": ds.descriptor_names,
            }
            try:
                joblib.dump(xgb_bundle, self.checkpoint_dir / xgb_file, compress=3)
            except Exception as exc:
                LOGGER.warning("Could not cache %s: %s", xgb_file, exc)

        return rf_bundle, xgb_bundle

    def _load_gnn(
        self, filename: str
    ) -> tuple[NeuroGINEModel | None, np.ndarray | None, np.ndarray | None]:
        path = self.checkpoint_dir / filename
        if not path.exists():
            LOGGER.warning("GNN checkpoint missing (%s); falling back to RF+XGB ensemble.", path)
            return None, None, None
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            cfg = GINEConfig(**ckpt["config"])
            model = NeuroGINEModel(cfg).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            mean = np.asarray(ckpt["desc_scaler_mean"], dtype=np.float32)
            scale = np.asarray(ckpt["desc_scaler_scale"], dtype=np.float32)
            return model, mean, scale
        except Exception as exc:
            LOGGER.warning("Failed to load GNN checkpoint %s (%s); using RF+XGB fallback.", path, exc)
            return None, None, None

    def _prepare_mol_and_features(
        self, smiles: str
    ) -> tuple[str, Chem.Mol, np.ndarray, np.ndarray, dict[str, float]]:
        """Standardize SMILES and extract Morgan FP + 8 descriptors."""
        canon_smiles, mol, err = self.cleaner.standardize_smiles(smiles)
        if err is not None or canon_smiles is None or mol is None:
            raise ValueError(f"Invalid or unparseable SMILES: {smiles!r} ({err})")

        desc_dict = self.extractor.compute_descriptors(mol)
        desc_vec = np.array([desc_dict[col] for col in DESCRIPTOR_COLUMNS], dtype=np.float32)
        fp_vec = self.extractor.compute_morgan_fp(mol).astype(np.float32)
        return canon_smiles, mol, fp_vec, desc_vec, desc_dict

    def _get_morgan_bit_info(self, mol: Chem.Mol) -> tuple[np.ndarray, dict[int, tuple[tuple[int, int], ...]]]:
        """Compute 2048-bit Morgan FP and atom-environment bitInfo dictionary."""
        bit_info: dict[int, tuple[tuple[int, int], ...]] = {}
        try:
            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            fp_gen = self.extractor._morgan_gen
            if fp_gen is not None:
                fp_arr = fp_gen.GetFingerprintAsNumPy(mol, additionalOutput=ao).astype(np.float32)
                bit_info = dict(ao.GetBitInfoMap())
                return fp_arr, bit_info
        except TypeError:
            pass

        bv = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS, bitInfo=bit_info
        )
        fp_arr = np.zeros((MORGAN_NBITS,), dtype=np.float32)
        from rdkit import DataStructs

        DataStructs.ConvertToNumpyArray(bv, fp_arr)
        return fp_arr, bit_info

    def predict_molecule(self, smiles: str) -> dict[str, Any]:
        """Run single-molecule ensemble inference for BBB+ probability, pIC50, and ADMET."""
        results = self.predict_batch([smiles])
        if not results or not results[0]["valid"]:
            raise ValueError(f"Invalid SMILES provided to predict_molecule: {smiles!r}")
        return results[0]

    def predict_batch(self, smiles_list: list[str]) -> list[dict[str, Any]]:
        """Vectorized/batched ensemble inference for a list of SMILES strings."""
        parsed: list[tuple[int, str, Chem.Mol, np.ndarray, np.ndarray, dict[str, float]]] = []
        outputs: list[dict[str, Any] | None] = [None] * len(smiles_list)

        for idx, smi in enumerate(smiles_list):
            try:
                canon, mol, fp_vec, desc_vec, desc_dict = self._prepare_mol_and_features(smi)
                parsed.append((idx, canon, mol, fp_vec, desc_vec, desc_dict))
            except Exception:
                outputs[idx] = {"input_smiles": smi, "valid": False}

        if not parsed:
            return [o for o in outputs if o is not None]

        fps = np.vstack([item[3] for item in parsed]).astype(np.float32)
        descs = np.vstack([item[4] for item in parsed]).astype(np.float32)

        desc_bbbp = self.bbbp_scaler.transform(descs).astype(np.float32)
        desc_ache = self.ache_scaler.transform(descs).astype(np.float32)

        X_bbbp = np.hstack([fps, desc_bbbp]).astype(np.float32)
        X_ache = np.hstack([fps, desc_ache]).astype(np.float32)

        bbb_rf = self.rf_bbbp.predict_proba(X_bbbp)[:, 1]
        bbb_xgb = self.xgb_bbbp.predict_proba(X_bbbp)[:, 1]
        pic50_rf = self.rf_ache.predict(X_ache)
        pic50_xgb = self.xgb_ache.predict(X_ache)

        bbb_gnn = None
        if self.gnn_bbbp is not None and self.gnn_bbbp_mean is not None and self.gnn_bbbp_scale is not None:
            desc_gnn_bbbp = (descs - self.gnn_bbbp_mean) / np.where(
                self.gnn_bbbp_scale == 0, 1.0, self.gnn_bbbp_scale
            )
            graphs_bbbp = [
                smiles_to_pyg_data(item[1], descriptors=d) for item, d in zip(parsed, desc_gnn_bbbp)
            ]
            bbb_gnn = self._batched_gnn_forward(self.gnn_bbbp, graphs_bbbp, apply_sigmoid=True)

        pic50_gnn = None
        if self.gnn_ache is not None and self.gnn_ache_mean is not None and self.gnn_ache_scale is not None:
            desc_gnn_ache = (descs - self.gnn_ache_mean) / np.where(
                self.gnn_ache_scale == 0, 1.0, self.gnn_ache_scale
            )
            graphs_ache = [
                smiles_to_pyg_data(item[1], descriptors=d) for item, d in zip(parsed, desc_gnn_ache)
            ]
            pic50_gnn = self._batched_gnn_forward(self.gnn_ache, graphs_ache, apply_sigmoid=False)

        for i, (orig_idx, canon, _, _, _, desc_dict) in enumerate(parsed):
            bbb_components = [bbb_rf[i], bbb_xgb[i]] + ([bbb_gnn[i]] if bbb_gnn is not None else [])
            pic50_components = [pic50_rf[i], pic50_xgb[i]] + ([pic50_gnn[i]] if pic50_gnn is not None else [])
            bbb_ens = float(np.mean(bbb_components))
            pic50_ens = float(np.mean(pic50_components))
            ic50_nm = float(10.0 ** (9.0 - pic50_ens))
            violations, lipinski_pass = _compute_lipinski_status(desc_dict)

            outputs[orig_idx] = {
                "valid": True,
                "input_smiles": smiles_list[orig_idx],
                "canonical_smiles": canon,
                "bbb_prob_rf": float(bbb_rf[i]),
                "bbb_prob_xgb": float(bbb_xgb[i]),
                "bbb_prob_gnn": float(bbb_gnn[i]) if bbb_gnn is not None else float("nan"),
                "bbb_prob_ensemble": bbb_ens,
                "pic50_rf": float(pic50_rf[i]),
                "pic50_xgb": float(pic50_xgb[i]),
                "pic50_gnn": float(pic50_gnn[i]) if pic50_gnn is not None else float("nan"),
                "pic50_ensemble": pic50_ens,
                "ic50_nm": ic50_nm,
                "descriptors": desc_dict,
                "lipinski_violations": int(violations),
                "lipinski_pass": bool(lipinski_pass),
            }

        return [o for o in outputs if o is not None]

    @torch.no_grad()
    def _batched_gnn_forward(
        self,
        model: NeuroGINEModel,
        graphs: list[Any],
        apply_sigmoid: bool,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Run batched PyG inference returning 1D NumPy predictions."""
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        preds: list[np.ndarray] = []
        for batch in loader:
            batch = batch.to(self.device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
            if apply_sigmoid:
                out = torch.sigmoid(out)
            preds.append(out.view(-1).cpu().numpy())
        return np.concatenate(preds)

    def _compute_xgb_tree_shap(
        self,
        xgb_model: Any,
        shap_explainer: Any | None,
        X_row: np.ndarray,
    ) -> np.ndarray:
        """Compute exact 2056-dim TreeSHAP feature contributions for one molecule."""
        if shap_explainer is not None:
            sv = shap_explainer.shap_values(X_row)
            if isinstance(sv, list):
                sv = sv[-1]
            sv_arr = np.asarray(sv, dtype=np.float64)
            if sv_arr.ndim == 3:
                sv_arr = sv_arr[:, :, -1]
            return sv_arr.reshape(-1)

        # Fallback: XGBoost built-in exact C++ TreeSHAP
        booster = xgb_model.get_booster()
        contribs = booster.predict(xgb.DMatrix(X_row), pred_contribs=True)
        return np.asarray(contribs[0, :-1], dtype=np.float64)

    def _compute_rf_active_bit_attribution(
        self,
        rf_model: Any,
        X_row: np.ndarray,
        active_bits: list[int],
        is_classifier: bool,
    ) -> dict[int, float]:
        """Measure Random Forest prediction drop when ablating each active Morgan bit."""
        if not active_bits:
            return {}

        base_pred = float(
            rf_model.predict_proba(X_row)[0, 1] if is_classifier else rf_model.predict(X_row)[0]
        )
        perturbed = np.repeat(X_row, len(active_bits), axis=0)
        for row_i, bit_idx in enumerate(active_bits):
            perturbed[row_i, bit_idx] = 0.0

        pert_preds = (
            rf_model.predict_proba(perturbed)[:, 1]
            if is_classifier
            else rf_model.predict(perturbed)
        )
        return {
            bit_idx: float(base_pred - pert_preds[row_i])
            for row_i, bit_idx in enumerate(active_bits)
        }

    def _map_bit_attributions_to_atoms(
        self,
        mol: Chem.Mol,
        bit_info: dict[int, tuple[tuple[int, int], ...]],
        xgb_shap: np.ndarray,
        rf_bit_deltas: dict[int, float],
    ) -> np.ndarray:
        """Map active Morgan FP bit attributions back to contributing atoms via bitInfo."""
        n_atoms = mol.GetNumAtoms()
        xgb_atom = np.zeros(n_atoms, dtype=np.float64)
        rf_atom = np.zeros(n_atoms, dtype=np.float64)

        for bit_idx, envs in bit_info.items():
            if not envs:
                continue
            xgb_val = float(xgb_shap[bit_idx])
            rf_val = float(rf_bit_deltas.get(bit_idx, 0.0))
            n_envs = len(envs)

            for center_atom, radius in envs:
                if radius == 0:
                    env_atoms = {center_atom}
                else:
                    bond_ids = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
                    env_atoms = {center_atom}
                    for b_idx in bond_ids:
                        bond = mol.GetBondWithIdx(b_idx)
                        env_atoms.add(bond.GetBeginAtomIdx())
                        env_atoms.add(bond.GetEndAtomIdx())

                share = 1.0 / (n_envs * max(len(env_atoms), 1))
                for atom_idx in env_atoms:
                    xgb_atom[atom_idx] += xgb_val * share
                    rf_atom[atom_idx] += rf_val * share

        return 0.5 * _normalize_signed(xgb_atom) + 0.5 * _normalize_signed(rf_atom)

    def _compute_gnn_atom_attribution(
        self,
        model: NeuroGINEModel,
        canon_smiles: str,
        desc_scaled: np.ndarray,
        apply_sigmoid: bool,
    ) -> np.ndarray:
        """Combine Input*Gradient saliency and batched atom occlusion delta for GINEConv."""
        data = smiles_to_pyg_data(canon_smiles, descriptors=desc_scaled)
        n_atoms = data.x.size(0)
        if n_atoms == 0:
            return np.zeros(0, dtype=np.float64)

        # 1. Input x Gradient saliency
        data_grad = data.clone().to(self.device)
        data_grad.x.requires_grad_(True)
        batch_vec = torch.zeros(n_atoms, dtype=torch.long, device=self.device)

        model.zero_grad()
        raw_out = model(data_grad.x, data_grad.edge_index, data_grad.edge_attr, batch_vec, data_grad.u)
        scalar_out = torch.sigmoid(raw_out)[0] if apply_sigmoid else raw_out[0]
        scalar_out.backward()
        grad_sal = (data_grad.x * data_grad.x.grad).sum(dim=-1).detach().cpu().numpy()

        # 2. Batched atom occlusion delta: y(G) - y(G without atom i features)
        base_val = float(scalar_out.detach().cpu().item())
        masked_graphs = []
        for atom_idx in range(n_atoms):
            g_mask = data.clone()
            g_mask.x[atom_idx, :] = 0.0
            masked_graphs.append(g_mask)

        masked_preds = self._batched_gnn_forward(
            model, masked_graphs, apply_sigmoid=apply_sigmoid, batch_size=64
        )
        occlusion_delta = base_val - masked_preds

        return 0.5 * _normalize_signed(grad_sal) + 0.5 * _normalize_signed(occlusion_delta)

    @staticmethod
    def render_atom_heatmap_png(
        smiles: str,
        atom_weights: list[float],
        width: int = 500,
        height: int = 380,
    ) -> bytes:
        """Render a high-resolution 2D RDKit PNG heatmap (green=positive, red=negative)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        if len(atom_weights) != mol.GetNumAtoms():
            raise ValueError(f"Expected {mol.GetNumAtoms()} weights, got {len(atom_weights)}.")

        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.padding = 0.08
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol,
            [float(w) for w in atom_weights],
            draw2d=drawer,
            colorMap=plt.get_cmap("RdYlGn"),
            contourLines=10,
            alpha=0.45,
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    @staticmethod
    def render_atom_heatmap_svg(
        smiles: str,
        atom_weights: list[float],
        width: int = 500,
        height: int = 380,
    ) -> str:
        """Render a 2D RDKit SVG atom-attribution heatmap."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        if len(atom_weights) != mol.GetNumAtoms():
            raise ValueError(f"Expected {mol.GetNumAtoms()} weights, got {len(atom_weights)}.")

        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.padding = 0.08
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol,
            [float(w) for w in atom_weights],
            draw2d=drawer,
            colorMap=plt.get_cmap("RdYlGn"),
            contourLines=10,
            alpha=0.45,
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def explain_molecule(self, smiles: str, width: int = 500, height: int = 380) -> AttributionResult:
        """Compute ensemble prediction, dual XAI atom attributions, and 2D heatmaps."""
        pred = self.predict_molecule(smiles)
        canon_smiles = pred["canonical_smiles"]
        mol = Chem.MolFromSmiles(canon_smiles)
        assert mol is not None

        fp_arr, bit_info = self._get_morgan_bit_info(mol)
        active_bits = sorted(bit_info.keys())

        desc_vec = np.array(
            [pred["descriptors"][col] for col in DESCRIPTOR_COLUMNS], dtype=np.float32
        ).reshape(1, -1)

        X_bbbp = np.hstack([fp_arr.reshape(1, -1), self.bbbp_scaler.transform(desc_vec)]).astype(
            np.float32
        )
        X_ache = np.hstack([fp_arr.reshape(1, -1), self.ache_scaler.transform(desc_vec)]).astype(
            np.float32
        )

        # Method A: TreeSHAP + RF active-bit perturbation mapped via bitInfo
        xgb_shap_bbb = self._compute_xgb_tree_shap(self.xgb_bbbp, self._shap_xgb_bbbp, X_bbbp)
        rf_deltas_bbb = self._compute_rf_active_bit_attribution(
            self.rf_bbbp, X_bbbp, active_bits, is_classifier=True
        )
        shap_atoms_bbb = self._map_bit_attributions_to_atoms(mol, bit_info, xgb_shap_bbb, rf_deltas_bbb)

        xgb_shap_ache = self._compute_xgb_tree_shap(self.xgb_ache, self._shap_xgb_ache, X_ache)
        rf_deltas_ache = self._compute_rf_active_bit_attribution(
            self.rf_ache, X_ache, active_bits, is_classifier=False
        )
        shap_atoms_ache = self._map_bit_attributions_to_atoms(
            mol, bit_info, xgb_shap_ache, rf_deltas_ache
        )

        desc_shap_bbb = {
            col: float(xgb_shap_bbb[MORGAN_NBITS + idx])
            for idx, col in enumerate(DESCRIPTOR_COLUMNS)
        }
        desc_shap_ache = {
            col: float(xgb_shap_ache[MORGAN_NBITS + idx])
            for idx, col in enumerate(DESCRIPTOR_COLUMNS)
        }

        # Method B: GINEConv Input*Gradient + Atom Occlusion Saliency
        if self.gnn_bbbp is not None and self.gnn_bbbp_mean is not None and self.gnn_bbbp_scale is not None:
            desc_gnn_bbbp = (desc_vec[0] - self.gnn_bbbp_mean) / np.where(
                self.gnn_bbbp_scale == 0, 1.0, self.gnn_bbbp_scale
            )
            gnn_atoms_bbb = self._compute_gnn_atom_attribution(
                self.gnn_bbbp, canon_smiles, desc_gnn_bbbp, apply_sigmoid=True
            )
            fused_bbb = _normalize_signed(
                0.5 * _normalize_signed(shap_atoms_bbb) + 0.5 * _normalize_signed(gnn_atoms_bbb)
            ).tolist()
        else:
            gnn_atoms_bbb = np.zeros(mol.GetNumAtoms(), dtype=np.float64)
            fused_bbb = _normalize_signed(shap_atoms_bbb).tolist()

        if self.gnn_ache is not None and self.gnn_ache_mean is not None and self.gnn_ache_scale is not None:
            desc_gnn_ache = (desc_vec[0] - self.gnn_ache_mean) / np.where(
                self.gnn_ache_scale == 0, 1.0, self.gnn_ache_scale
            )
            gnn_atoms_ache = self._compute_gnn_atom_attribution(
                self.gnn_ache, canon_smiles, desc_gnn_ache, apply_sigmoid=False
            )
            fused_ache = _normalize_signed(
                0.5 * _normalize_signed(shap_atoms_ache) + 0.5 * _normalize_signed(gnn_atoms_ache)
            ).tolist()
        else:
            gnn_atoms_ache = np.zeros(mol.GetNumAtoms(), dtype=np.float64)
            fused_ache = _normalize_signed(shap_atoms_ache).tolist()

        png_bbb = self.render_atom_heatmap_png(canon_smiles, fused_bbb, width=width, height=height)
        png_ache = self.render_atom_heatmap_png(canon_smiles, fused_ache, width=width, height=height)
        svg_bbb = self.render_atom_heatmap_svg(canon_smiles, fused_bbb, width=width, height=height)
        svg_ache = self.render_atom_heatmap_svg(canon_smiles, fused_ache, width=width, height=height)

        return AttributionResult(
            canonical_smiles=canon_smiles,
            prediction=pred,
            atom_weights_bbb=[float(w) for w in fused_bbb],
            atom_weights_ache=[float(w) for w in fused_ache],
            shap_atom_weights_bbb=[float(w) for w in _normalize_signed(shap_atoms_bbb)],
            gnn_atom_weights_bbb=[float(w) for w in _normalize_signed(gnn_atoms_bbb)],
            shap_atom_weights_ache=[float(w) for w in _normalize_signed(shap_atoms_ache)],
            gnn_atom_weights_ache=[float(w) for w in _normalize_signed(gnn_atoms_ache)],
            descriptor_shap_bbb=desc_shap_bbb,
            descriptor_shap_ache=desc_shap_ache,
            png_bbb=png_bbb,
            png_ache=png_ache,
            svg_bbb=svg_bbb,
            svg_ache=svg_ache,
        )


def save_dual_xai_figure(
    result: AttributionResult,
    output_path: Path | str,
    molecule_title: str = "Donepezil",
) -> Path:
    """Save a 2-panel side-by-side BBB+ and AChE pIC50 XAI attribution figure (300 DPI)."""
    from io import BytesIO

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_bbb = Image.open(BytesIO(result.png_bbb))
    img_ache = Image.open(BytesIO(result.png_ache))
    pred = result.prediction

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), dpi=300)
    axes[0].imshow(img_bbb)
    axes[0].axis("off")
    axes[0].set_title(
        f"{molecule_title} — BBB+ Permeability XAI\n"
        f"Ensemble P(BBB+) = {pred['bbb_prob_ensemble']:.3f} "
        f"(RF={pred['bbb_prob_rf']:.2f}, XGB={pred['bbb_prob_xgb']:.2f}, GNN={pred['bbb_prob_gnn']:.2f})",
        fontsize=10.0,
        fontweight="bold",
        pad=10,
    )

    axes[1].imshow(img_ache)
    axes[1].axis("off")
    axes[1].set_title(
        f"{molecule_title} — Human AChE pIC50 XAI\n"
        f"Ensemble pIC50 = {pred['pic50_ensemble']:.2f} "
        f"(IC50 ≈ {pred['ic50_nm']:.1f} nM | QED={pred['descriptors']['QED']:.2f}, SA={pred['descriptors']['SA_Score']:.2f})",
        fontsize=10.0,
        fontweight="bold",
        pad=10,
    )

    fig.suptitle(
        "NeuroGen-XAI Dual Attribution (TreeSHAP + GINEConv Saliency) | Green = Positive, Red = Limiting",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved Dual-XAI figure -> %s", out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    explainer = NeuroGenExplainer()
    attr = explainer.explain_molecule(DONEPEZIL_SMILES)
    save_dual_xai_figure(
        attr,
        output_path=Path("reports/figures/xai_donepezil_attribution.png"),
        molecule_title="Donepezil",
    )
    p = attr.prediction
    print(
        f"Donepezil XAI verified: P(BBB+)={p['bbb_prob_ensemble']:.4f}, "
        f"pIC50={p['pic50_ensemble']:.3f} (IC50={p['ic50_nm']:.2f} nM), "
        f"QED={p['descriptors']['QED']:.3f}, SA={p['descriptors']['SA_Score']:.3f}, "
        f"Lipinski={'PASS' if p['lipinski_pass'] else 'FAIL'}"
    )
