"""Phase 1: Data Curation, Standardization, Featurization, and Bemis-Murcko Scaffold Splitting.

This script builds the foundational datasets for NeuroGen-XAI:
  1. Downloads MoleculeNet BBBP (Blood-Brain Barrier Penetration) dataset.
  2. Fetches Human Acetylcholinesterase (CHEMBL220) IC50 bioactivity records
     from the ChEMBL API and converts IC50 (nM) to pIC50.
  3. Cleans and standardizes all molecules via RDKit (invalid SMILES removal,
     salt/fragment stripping, charge neutralization, canonicalization, deduplication).
  4. Computes 8 CNS physicochemical descriptors (MW, LogP, TPSA, HBD, HBA,
     RotBonds, QED, SA_Score) and 2048-bit Morgan fingerprints (radius 2).
  5. Splits both datasets into 80% Train / 10% Validation / 10% Test partitions
     using deterministic Bemis-Murcko Scaffold Splitting with zero scaffold leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, MolSurf, QED, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from urllib3.util.retry import Retry


LOGGER = logging.getLogger("neurogen_xai.data_pipeline")

BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
CHEMBL_ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
CHEMBL_TARGET_ID = "CHEMBL220"

DESCRIPTOR_COLUMNS = [
    "MW",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
    "QED",
    "SA_Score",
]
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
ACYCLIC_SCAFFOLD_LABEL = "[ACYCLIC]"


def _load_sascorer() -> Any | None:
    """Load RDKit's Ertl & Schuffenhauer SA_Score module from RDContribDir."""
    sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if os.path.isdir(sa_dir) and sa_dir not in sys.path:
        sys.path.insert(0, sa_dir)
    try:
        import sascorer  # type: ignore[import-not-found]

        return sascorer
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning(
            "Could not load RDKit Contrib sascorer (%s); falling back to ring/spiro complexity heuristic.",
            exc,
        )
        return None


_SASCORER = _load_sascorer()


def _create_http_session(retries: int = 4, backoff_factor: float = 1.0) -> requests.Session:
    """Create a requests.Session configured with exponential backoff retries."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "NeuroGen-XAI/0.1 (Computational Drug Discovery)"})
    return session


@dataclass
class CleaningReport:
    """Tracks molecule retention and filtering statistics during RDKit curation."""

    dataset_name: str
    raw_records: int
    invalid_smiles_dropped: int = 0
    empty_mol_dropped: int = 0
    standardization_failed_dropped: int = 0
    duplicates_collapsed: int = 0
    conflicting_or_noisy_dropped: int = 0
    final_unique_molecules: int = 0


def fetch_bbbp_dataset(
    raw_dir: Path,
    force_download: bool = False,
    timeout: int = 60,
) -> pd.DataFrame:
    """Download MoleculeNet BBBP dataset and return raw DataFrame."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "bbbp_raw.csv"

    if raw_path.exists() and not force_download:
        LOGGER.info("Using cached BBBP raw dataset at %s", raw_path)
        return pd.read_csv(raw_path)

    LOGGER.info("Downloading MoleculeNet BBBP dataset from %s", BBBP_URL)
    session = _create_http_session()
    response = session.get(BBBP_URL, timeout=timeout)
    response.raise_for_status()
    raw_path.write_bytes(response.content)

    df = pd.read_csv(raw_path)
    LOGGER.info("Downloaded BBBP dataset: %d raw rows -> %s", len(df), raw_path)
    return df


def fetch_chembl220_dataset(
    raw_dir: Path,
    force_download: bool = False,
    max_records: int | None = None,
    use_chembl_client: bool = False,
    page_size: int = 1000,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch Human Acetylcholinesterase (CHEMBL220) IC50 records from ChEMBL."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "chembl220_raw.csv"

    if raw_path.exists() and not force_download:
        LOGGER.info("Using cached CHEMBL220 raw dataset at %s", raw_path)
        return pd.read_csv(raw_path)

    if use_chembl_client:
        records = _fetch_chembl220_via_client(max_records=max_records)
    else:
        records = _fetch_chembl220_via_rest(
            max_records=max_records,
            page_size=page_size,
            timeout=timeout,
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError(
            "Fetched 0 bioactivity records for CHEMBL220. "
            "Check network connectivity or ChEMBL API availability."
        )

    df.to_csv(raw_path, index=False)
    LOGGER.info("Fetched CHEMBL220 dataset: %d raw IC50 rows -> %s", len(df), raw_path)
    return df


def _fetch_chembl220_via_client(max_records: int | None = None) -> list[dict[str, Any]]:
    """Optional fetch path using chembl_webresource_client."""
    from chembl_webresource_client.new_client import new_client  # type: ignore[import-not-found]

    LOGGER.info("Fetching %s IC50 records via chembl_webresource_client...", CHEMBL_TARGET_ID)
    activity_api = new_client.activity
    query = activity_api.filter(
        target_chembl_id=CHEMBL_TARGET_ID,
        standard_type="IC50",
        standard_units="nM",
        standard_relation="=",
    ).only(
        [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_type",
            "standard_relation",
            "standard_value",
            "standard_units",
            "pchembl_value",
            "assay_chembl_id",
            "document_chembl_id",
        ]
    )

    records: list[dict[str, Any]] = []
    for item in tqdm(query, desc="ChEMBL220 (client)", unit="rec"):
        records.append(_extract_chembl_record(item))
        if max_records is not None and len(records) >= max_records:
            break
    return records


def _fetch_chembl220_via_rest(
    max_records: int | None = None,
    page_size: int = 1000,
    timeout: int = 60,
) -> list[dict[str, Any]]:
    """Fetch CHEMBL220 IC50 records using paginated ChEMBL REST API calls."""
    LOGGER.info("Fetching %s IC50 bioactivity records from ChEMBL REST API...", CHEMBL_TARGET_ID)
    session = _create_http_session()
    offset = 0
    records: list[dict[str, Any]] = []
    pbar: tqdm | None = None

    try:
        while True:
            limit = page_size
            if max_records is not None:
                remaining = max_records - len(records)
                if remaining <= 0:
                    break
                limit = min(page_size, remaining)

            params = {
                "target_chembl_id": CHEMBL_TARGET_ID,
                "standard_type": "IC50",
                "standard_units": "nM",
                "standard_relation": "=",
                "limit": limit,
                "offset": offset,
            }
            response = session.get(CHEMBL_ACTIVITY_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()

            activities = payload.get("activities", [])
            page_meta = payload.get("page_meta", {})
            total_count = page_meta.get("total_count")

            if pbar is None:
                target_total = min(total_count, max_records) if (total_count and max_records) else total_count
                pbar = tqdm(total=target_total, desc="ChEMBL220 (REST)", unit="rec")

            if not activities:
                break

            for item in activities:
                records.append(_extract_chembl_record(item))

            pbar.update(len(activities))

            if max_records is not None and len(records) >= max_records:
                records = records[:max_records]
                break

            if not page_meta.get("next"):
                break

            offset += len(activities)
            time.sleep(0.05)
    finally:
        if pbar is not None:
            pbar.close()

    return records


def _extract_chembl_record(item: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized bioactivity fields from a ChEMBL API activity object."""
    return {
        "molecule_chembl_id": item.get("molecule_chembl_id"),
        "smiles": item.get("canonical_smiles"),
        "standard_type": item.get("standard_type"),
        "standard_relation": item.get("standard_relation"),
        "standard_value": item.get("standard_value"),
        "standard_units": item.get("standard_units"),
        "pchembl_value": item.get("pchembl_value"),
        "assay_chembl_id": item.get("assay_chembl_id"),
        "document_chembl_id": item.get("document_chembl_id"),
    }


class MolCleaner:
    """RDKit standardization utility: salt stripping, neutralization, canonicalization."""

    def __init__(self) -> None:
        self.largest_fragment = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        self.uncharger = rdMolStandardize.Uncharger()

    def standardize_smiles(self, raw_smiles: Any) -> tuple[str | None, Chem.Mol | None, str | None]:
        """Clean and canonicalize a raw SMILES string.

        Returns:
            Tuple of (canonical_smiles, standardized_mol, failure_reason).
        """
        if not isinstance(raw_smiles, str) or not raw_smiles.strip():
            return None, None, "invalid_smiles"

        mol = Chem.MolFromSmiles(raw_smiles.strip())
        if mol is None:
            return None, None, "invalid_smiles"
        if mol.GetNumHeavyAtoms() == 0:
            return None, None, "empty_mol"

        try:
            # 1. Strip salts / counterions by selecting largest organic fragment
            parent_mol = self.largest_fragment.choose(mol)
            if parent_mol is None or parent_mol.GetNumHeavyAtoms() == 0:
                frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if not frags:
                    return None, None, "empty_mol"
                parent_mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

            # 2. Neutralize formal charges where chemically valid
            neutral_mol = self.uncharger.uncharge(parent_mol)
            Chem.SanitizeMol(neutral_mol)

            if neutral_mol.GetNumHeavyAtoms() == 0:
                return None, None, "empty_mol"

            # 3. Canonicalize (preserving stereochemistry) and verify round-trip parsing
            canonical_smiles = Chem.MolToSmiles(
                neutral_mol,
                canonical=True,
                isomericSmiles=True,
            )
            verified_mol = Chem.MolFromSmiles(canonical_smiles)
            if verified_mol is None or verified_mol.GetNumHeavyAtoms() == 0:
                return None, None, "standardization_failed"

            return canonical_smiles, verified_mol, None
        except Exception:
            return None, None, "standardization_failed"


def clean_bbbp_dataframe(
    raw_df: pd.DataFrame,
    cleaner: MolCleaner,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean MoleculeNet BBBP data and resolve duplicate / conflicting SMILES."""
    report = CleaningReport(dataset_name="BBBP", raw_records=len(raw_df))
    cleaned_rows: list[dict[str, Any]] = []

    RDLogger.DisableLog("rdApp.*")
    try:
        for row in raw_df.itertuples(index=False):
            smiles_val = getattr(row, "smiles", None)
            label_val = getattr(row, "p_np", None)
            name_val = getattr(row, "name", "")

            if pd.isna(label_val):
                report.invalid_smiles_dropped += 1
                continue

            try:
                label_int = int(label_val)
            except (TypeError, ValueError):
                report.invalid_smiles_dropped += 1
                continue

            if label_int not in (0, 1):
                report.invalid_smiles_dropped += 1
                continue

            canon_smiles, _, err = cleaner.standardize_smiles(smiles_val)
            if err == "invalid_smiles":
                report.invalid_smiles_dropped += 1
                continue
            if err == "empty_mol":
                report.empty_mol_dropped += 1
                continue
            if err == "standardization_failed" or canon_smiles is None:
                report.standardization_failed_dropped += 1
                continue

            cleaned_rows.append(
                {
                    "canonical_smiles": canon_smiles,
                    "p_np": label_int,
                    "compound_name": str(name_val) if pd.notna(name_val) else "",
                }
            )
    finally:
        RDLogger.EnableLog("rdApp.*")

    df = pd.DataFrame(cleaned_rows)
    if df.empty:
        raise RuntimeError("No valid molecules remained in BBBP after standardization.")

    # Deduplicate by canonical_smiles; drop compounds with conflicting binary labels
    dedup_rows: list[dict[str, Any]] = []
    for canon_smiles, group in df.groupby("canonical_smiles", sort=True):
        unique_labels = group["p_np"].unique()
        if len(unique_labels) > 1:
            report.conflicting_or_noisy_dropped += len(group)
            continue
        if len(group) > 1:
            report.duplicates_collapsed += len(group) - 1
        dedup_rows.append(
            {
                "canonical_smiles": canon_smiles,
                "p_np": int(unique_labels[0]),
                "compound_name": group["compound_name"].iloc[0],
            }
        )

    out_df = pd.DataFrame(dedup_rows).reset_index(drop=True)
    report.final_unique_molecules = len(out_df)
    return out_df, report


def clean_chembl220_dataframe(
    raw_df: pd.DataFrame,
    cleaner: MolCleaner,
    max_replicate_std: float = 1.0,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Filter CHEMBL220 IC50 records, convert to pIC50, standardize, and deduplicate."""
    report = CleaningReport(dataset_name="CHEMBL220", raw_records=len(raw_df))

    # Filter strictly to exact IC50 measurements in nM with positive values
    df = raw_df.copy()
    df["standard_value_num"] = pd.to_numeric(df["standard_value"], errors="coerce")
    valid_mask = (
        (df["standard_type"] == "IC50")
        & (df["standard_units"] == "nM")
        & (df["standard_relation"].astype(str).str.strip() == "=")
        & df["standard_value_num"].notna()
        & (df["standard_value_num"] > 0.0)
    )
    filtered_df = df.loc[valid_mask].copy()
    pre_filter_dropped = len(df) - len(filtered_df)
    if pre_filter_dropped > 0:
        LOGGER.info(
            "CHEMBL220 pre-filter removed %d non-exact/non-nM/non-positive IC50 rows.",
            pre_filter_dropped,
        )

    cleaned_rows: list[dict[str, Any]] = []
    RDLogger.DisableLog("rdApp.*")
    try:
        for row in filtered_df.itertuples(index=False):
            smiles_val = getattr(row, "smiles", None)
            ic50_nm = float(getattr(row, "standard_value_num"))
            chembl_id = getattr(row, "molecule_chembl_id", "")

            canon_smiles, _, err = cleaner.standardize_smiles(smiles_val)
            if err == "invalid_smiles":
                report.invalid_smiles_dropped += 1
                continue
            if err == "empty_mol":
                report.empty_mol_dropped += 1
                continue
            if err == "standardization_failed" or canon_smiles is None:
                report.standardization_failed_dropped += 1
                continue

            pic50 = 9.0 - math.log10(ic50_nm)
            cleaned_rows.append(
                {
                    "canonical_smiles": canon_smiles,
                    "pIC50": pic50,
                    "ic50_nm": ic50_nm,
                    "molecule_chembl_id": str(chembl_id) if pd.notna(chembl_id) else "",
                }
            )
    finally:
        RDLogger.EnableLog("rdApp.*")

    clean_df = pd.DataFrame(cleaned_rows)
    if clean_df.empty:
        raise RuntimeError("No valid molecules remained in CHEMBL220 after standardization.")

    # Aggregate replicate measurements by median pIC50; drop noisy assays (std > 1.0)
    dedup_rows: list[dict[str, Any]] = []
    for canon_smiles, group in clean_df.groupby("canonical_smiles", sort=True):
        if len(group) > 1:
            rep_std = float(group["pIC50"].std(ddof=1))
            if rep_std > max_replicate_std:
                report.conflicting_or_noisy_dropped += len(group)
                continue
            report.duplicates_collapsed += len(group) - 1

        median_pic50 = float(group["pIC50"].median())
        median_ic50 = float(group["ic50_nm"].median())
        dedup_rows.append(
            {
                "canonical_smiles": canon_smiles,
                "pIC50": round(median_pic50, 6),
                "ic50_nm": round(median_ic50, 6),
                "n_replicates": int(len(group)),
                "molecule_chembl_id": group["molecule_chembl_id"].iloc[0],
            }
        )

    out_df = pd.DataFrame(dedup_rows).reset_index(drop=True)
    report.final_unique_molecules = len(out_df)
    return out_df, report


class FeatureExtractor:
    """Computes physicochemical descriptors, SA Score, and 2048-bit Morgan fingerprints."""

    def __init__(self, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_NBITS) -> None:
        self.radius = radius
        self.n_bits = n_bits
        if hasattr(rdFingerprintGenerator, "GetMorganGenerator"):
            self._morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius,
                fpSize=self.n_bits,
            )
        else:  # pragma: no cover - legacy RDKit fallback
            self._morgan_gen = None

    @staticmethod
    def compute_sa_score(mol: Chem.Mol) -> float:
        """Compute Synthetic Accessibility Score (1=easy, 10=difficult)."""
        if _SASCORER is not None:
            return float(_SASCORER.calculateScore(mol))

        # Fallback complexity heuristic if RDKit Contrib SA_Score is unavailable
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        num_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        heavy_atoms = max(mol.GetNumHeavyAtoms(), 1)
        heuristic = 1.5 + 0.08 * heavy_atoms + 0.35 * num_rings + 0.5 * num_chiral
        return float(min(max(heuristic, 1.0), 10.0))

    def compute_descriptors(self, mol: Chem.Mol) -> dict[str, float]:
        """Compute the 8 CNS physicochemical and medicinal chemistry descriptors."""
        return {
            "MW": float(Descriptors.MolWt(mol)),
            "LogP": float(Crippen.MolLogP(mol)),
            "TPSA": float(MolSurf.TPSA(mol)),
            "HBD": float(Lipinski.NumHDonors(mol)),
            "HBA": float(Lipinski.NumHAcceptors(mol)),
            "RotBonds": float(Lipinski.NumRotatableBonds(mol)),
            "QED": float(QED.qed(mol)),
            "SA_Score": self.compute_sa_score(mol),
        }

    def compute_morgan_fp(self, mol: Chem.Mol) -> np.ndarray:
        """Compute 2048-bit radius-2 Morgan fingerprint as uint8 NumPy array."""
        if self._morgan_gen is not None:
            return self._morgan_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)

        from rdkit.Chem import AllChem  # pragma: no cover

        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr

    def featurize_dataframe(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Compute descriptors, Bemis-Murcko scaffolds, and Morgan FPs for a dataset."""
        desc_records: list[dict[str, float]] = []
        scaffolds: list[str] = []
        fps: list[np.ndarray] = []

        for smi in tqdm(df["canonical_smiles"], desc="Featurizing", unit="mol"):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Unexpected invalid canonical SMILES during featurization: {smi}")

            desc_records.append(self.compute_descriptors(mol))
            fps.append(self.compute_morgan_fp(mol))

            scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )
            scaffolds.append(scaffold_smi if scaffold_smi else ACYCLIC_SCAFFOLD_LABEL)

        desc_df = pd.DataFrame(desc_records, columns=DESCRIPTOR_COLUMNS)
        enriched_df = df.copy()
        enriched_df["scaffold_smiles"] = scaffolds
        for col in DESCRIPTOR_COLUMNS:
            enriched_df[col] = desc_df[col].values

        X_desc = desc_df.to_numpy(dtype=np.float32)
        X_fp = np.vstack(fps).astype(np.uint8)
        return enriched_df, X_desc, X_fp


def bemis_murcko_scaffold_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    shuffle_ties: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train/val/test by Bemis-Murcko scaffold with zero leakage.

    Entire scaffold buckets are assigned monotonically so no scaffold appears in
    more than one partition.
    """
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(
            f"Split fractions must sum to 1.0 (got {train_frac + val_frac + test_frac:.4f})."
        )

    scaffold_to_indices: dict[str, list[int]] = {}
    for idx, scaffold in enumerate(df["scaffold_smiles"]):
        scaffold_to_indices.setdefault(str(scaffold), []).append(idx)

    if len(scaffold_to_indices) < 3 and len(df) >= 3:
        raise RuntimeError(
            f"Need at least 3 unique scaffolds for train/val/test split, found {len(scaffold_to_indices)}."
        )

    buckets = list(scaffold_to_indices.items())
    if shuffle_ties:
        rng = np.random.default_rng(seed)
        tie_order = rng.permutation(len(buckets))
        buckets = [buckets[i] for i in tie_order]
    buckets.sort(key=lambda item: (-len(item[1]), item[0]))

    n_total = len(df)
    train_cutoff = train_frac * n_total
    val_cutoff = (train_frac + val_frac) * n_total

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for _, indices in buckets:
        if len(train_idx) + len(indices) <= train_cutoff or not train_idx:
            train_idx.extend(indices)
        elif len(train_idx) + len(val_idx) + len(indices) <= val_cutoff or not val_idx:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    # Ensure test set is non-empty when sufficient unique scaffolds exist
    if not test_idx and len(val_idx) > 1:
        last_val_scaffold = str(df.iloc[val_idx[-1]]["scaffold_smiles"])
        moved = [i for i in val_idx if str(df.iloc[i]["scaffold_smiles"]) == last_val_scaffold]
        val_idx = [i for i in val_idx if i not in set(moved)]
        test_idx.extend(moved)

    train_scaffolds = set(df.iloc[train_idx]["scaffold_smiles"])
    val_scaffolds = set(df.iloc[val_idx]["scaffold_smiles"])
    test_scaffolds = set(df.iloc[test_idx]["scaffold_smiles"])

    assert len(train_scaffolds & val_scaffolds) == 0, "Scaffold leakage between train and val!"
    assert len(train_scaffolds & test_scaffolds) == 0, "Scaffold leakage between train and test!"
    assert len(val_scaffolds & test_scaffolds) == 0, "Scaffold leakage between val and test!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_total, "Split size mismatch!"

    return (
        np.array(sorted(train_idx), dtype=np.int64),
        np.array(sorted(val_idx), dtype=np.int64),
        np.array(sorted(test_idx), dtype=np.int64),
    )


def _compute_split_summary(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
) -> dict[str, Any]:
    """Compute target distribution and scaffold counts for a dataframe slice."""
    summary: dict[str, Any] = {
        "n_molecules": int(len(df)),
        "n_unique_scaffolds": int(df["scaffold_smiles"].nunique()),
    }
    if task_type == "classification":
        positives = int((df[target_col] == 1).sum())
        negatives = int((df[target_col] == 0).sum())
        summary["positives_bbb_plus"] = positives
        summary["negatives_bbb_minus"] = negatives
        summary["positive_rate"] = round(positives / max(len(df), 1), 4)
    else:
        summary["target_mean"] = round(float(df[target_col].mean()), 4)
        summary["target_std"] = round(float(df[target_col].std(ddof=1)), 4) if len(df) > 1 else 0.0
        summary["target_min"] = round(float(df[target_col].min()), 4)
        summary["target_max"] = round(float(df[target_col].max()), 4)
    return summary


def save_dataset_artifacts(
    name: str,
    df: pd.DataFrame,
    X_desc: np.ndarray,
    X_fp: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    target_col: str,
    task_type: str,
    cleaning_report: CleaningReport,
    processed_dir: Path,
) -> dict[str, Any]:
    """Save split CSVs, compressed NPZ feature arrays, and return structured stats."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    prefix = name.lower()

    split_labels = np.empty(len(df), dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    df_out = df.copy()
    df_out["split"] = split_labels

    train_df = df_out.iloc[train_idx].reset_index(drop=True)
    val_df = df_out.iloc[val_idx].reset_index(drop=True)
    test_df = df_out.iloc[test_idx].reset_index(drop=True)

    df_out.to_csv(processed_dir / f"{prefix}_cleaned.csv", index=False)
    train_df.to_csv(processed_dir / f"{prefix}_train.csv", index=False)
    val_df.to_csv(processed_dir / f"{prefix}_val.csv", index=False)
    test_df.to_csv(processed_dir / f"{prefix}_test.csv", index=False)

    y_dtype = np.int64 if task_type == "classification" else np.float32
    y_arr = df_out[target_col].to_numpy(dtype=y_dtype)

    np.savez_compressed(
        processed_dir / f"{prefix}_features.npz",
        X_fp=X_fp,
        X_desc=X_desc,
        descriptor_names=np.array(DESCRIPTOR_COLUMNS),
        y=y_arr,
        smiles=df_out["canonical_smiles"].to_numpy(dtype=str),
        scaffolds=df_out["scaffold_smiles"].to_numpy(dtype=str),
        split=df_out["split"].to_numpy(dtype=str),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    desc_stats = {
        col: {
            "mean": round(float(df_out[col].mean()), 4),
            "std": round(float(df_out[col].std(ddof=1)), 4) if len(df_out) > 1 else 0.0,
            "min": round(float(df_out[col].min()), 4),
            "max": round(float(df_out[col].max()), 4),
        }
        for col in DESCRIPTOR_COLUMNS
    }

    return {
        "cleaning_report": asdict(cleaning_report),
        "scaffold_overlap_check": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
        },
        "splits": {
            "overall": _compute_split_summary(df_out, target_col, task_type),
            "train": _compute_split_summary(train_df, target_col, task_type),
            "val": _compute_split_summary(val_df, target_col, task_type),
            "test": _compute_split_summary(test_df, target_col, task_type),
        },
        "descriptor_statistics": desc_stats,
        "fingerprint_shape": list(X_fp.shape),
        "descriptor_shape": list(X_desc.shape),
    }


def log_summary_table(stats: dict[str, Any]) -> None:
    """Print a clean, readable summary of cleaning and scaffold-split statistics."""
    LOGGER.info("=" * 78)
    LOGGER.info("NEUROGEN-XAI PHASE 1 DATASET & SCAFFOLD-SPLIT SUMMARY")
    LOGGER.info("=" * 78)
    for ds_name, ds_stats in stats.items():
        clean = ds_stats["cleaning_report"]
        splits = ds_stats["splits"]
        LOGGER.info("[%s]", ds_name.upper())
        LOGGER.info(
            "  Raw: %d | Invalid SMILES: -%d | Empty: -%d | StdErr: -%d | Duplicates: -%d | Conflicts/Noisy: -%d -> Final: %d",
            clean["raw_records"],
            clean["invalid_smiles_dropped"],
            clean["empty_mol_dropped"],
            clean["standardization_failed_dropped"],
            clean["duplicates_collapsed"],
            clean["conflicting_or_noisy_dropped"],
            clean["final_unique_molecules"],
        )
        LOGGER.info(
            "  Features: Morgan FP %s + Physicochemical Descriptors %s | Scaffold Overlap: 0",
            tuple(ds_stats["fingerprint_shape"]),
            tuple(ds_stats["descriptor_shape"]),
        )
        for split_name in ("train", "val", "test"):
            sp = splits[split_name]
            if "positive_rate" in sp:
                LOGGER.info(
                    "  %-5s : %4d mols (%5.1f%%) | %4d scaffolds | BBB+ rate: %.3f (%d+ / %d-)",
                    split_name.upper(),
                    sp["n_molecules"],
                    100.0 * sp["n_molecules"] / max(splits["overall"]["n_molecules"], 1),
                    sp["n_unique_scaffolds"],
                    sp["positive_rate"],
                    sp["positives_bbb_plus"],
                    sp["negatives_bbb_minus"],
                )
            else:
                LOGGER.info(
                    "  %-5s : %4d mols (%5.1f%%) | %4d scaffolds | pIC50: %.3f ± %.3f [%.2f, %.2f]",
                    split_name.upper(),
                    sp["n_molecules"],
                    100.0 * sp["n_molecules"] / max(splits["overall"]["n_molecules"], 1),
                    sp["n_unique_scaffolds"],
                    sp["target_mean"],
                    sp["target_std"],
                    sp["target_min"],
                    sp["target_max"],
                )
        desc_line = ", ".join(
            f"{col}={vals['mean']:.2f}±{vals['std']:.2f}"
            for col, vals in ds_stats["descriptor_statistics"].items()
        )
        LOGGER.info("  Descriptors: %s", desc_line)
    LOGGER.info("=" * 78)


def run_pipeline(
    data_dir: Path = Path("data"),
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    max_chembl_records: int | None = None,
    use_chembl_client: bool = False,
    force_download: bool = False,
) -> dict[str, Any]:
    """Execute the full Phase 1 curation, featurization, and scaffold-split pipeline."""
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    cleaner = MolCleaner()
    extractor = FeatureExtractor()

    # 1. BBBP Dataset
    bbbp_raw = fetch_bbbp_dataset(raw_dir=raw_dir, force_download=force_download)
    bbbp_clean, bbbp_report = clean_bbbp_dataframe(bbbp_raw, cleaner=cleaner)
    bbbp_feat_df, bbbp_desc, bbbp_fp = extractor.featurize_dataframe(bbbp_clean)
    bbbp_train, bbbp_val, bbbp_test = bemis_murcko_scaffold_split(
        bbbp_feat_df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    bbbp_stats = save_dataset_artifacts(
        name="bbbp",
        df=bbbp_feat_df,
        X_desc=bbbp_desc,
        X_fp=bbbp_fp,
        train_idx=bbbp_train,
        val_idx=bbbp_val,
        test_idx=bbbp_test,
        target_col="p_np",
        task_type="classification",
        cleaning_report=bbbp_report,
        processed_dir=processed_dir,
    )

    # 2. CHEMBL220 (Human Acetylcholinesterase) Dataset
    chembl_raw = fetch_chembl220_dataset(
        raw_dir=raw_dir,
        force_download=force_download,
        max_records=max_chembl_records,
        use_chembl_client=use_chembl_client,
    )
    chembl_clean, chembl_report = clean_chembl220_dataframe(chembl_raw, cleaner=cleaner)
    chembl_feat_df, chembl_desc, chembl_fp = extractor.featurize_dataframe(chembl_clean)
    chembl_train, chembl_val, chembl_test = bemis_murcko_scaffold_split(
        chembl_feat_df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    chembl_stats = save_dataset_artifacts(
        name="chembl220",
        df=chembl_feat_df,
        X_desc=chembl_desc,
        X_fp=chembl_fp,
        train_idx=chembl_train,
        val_idx=chembl_val,
        test_idx=chembl_test,
        target_col="pIC50",
        task_type="regression",
        cleaning_report=chembl_report,
        processed_dir=processed_dir,
    )

    all_stats = {
        "bbbp": bbbp_stats,
        "chembl220": chembl_stats,
    }
    stats_path = processed_dir / "dataset_statistics.json"
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    log_summary_table(all_stats)
    LOGGER.info("Saved dataset statistics JSON -> %s", stats_path)
    return all_stats


def parse_args() -> argparse.Namespace:
    """Parse command-line options for Phase 1 pipeline execution."""
    parser = argparse.ArgumentParser(
        description="NeuroGen-XAI Phase 1: Data Curation & Bemis-Murcko Scaffold-Split Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory containing raw/ and processed/ (default: data).",
    )
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train split fraction.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-chembl-records",
        type=int,
        default=None,
        help="Optional cap on ChEMBL220 IC50 records fetched (default: fetch all).",
    )
    parser.add_argument(
        "--use-chembl-client",
        action="store_true",
        help="Use chembl_webresource_client instead of direct ChEMBL REST pagination.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw BBBP and CHEMBL220 datasets even if cached.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        max_chembl_records=args.max_chembl_records,
        use_chembl_client=args.use_chembl_client,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
