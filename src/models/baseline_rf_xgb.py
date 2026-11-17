"""Baseline Random Forest and XGBoost pipelines for BBB permeability and AChE pIC50.

Phase 2A implementation:
  - Loads Phase 1 scaffold-split NPZ features (2048-bit Morgan FP + 8 descriptors).
  - Standardizes descriptors strictly on the training partition to avoid leakage.
  - Trains balanced Random Forest and XGBoost models for:
      * Task A: BBBP binary classification (ROC-AUC, PR-AUC, F1, MCC, Balanced Acc)
      * Task B: CHEMBL220 pIC50 regression (RMSE, MAE, R2, Spearman rho)
  - Persists fitted checkpoints + scalers via joblib to `models/checkpoints/`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


LOGGER = logging.getLogger("neurogen_xai.models.baseline")


@dataclass
class ScaffoldSplitDataset:
    """Container for scaffold-split fingerprint and standardized descriptor features."""

    name: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    X_desc_train_scaled: np.ndarray
    X_desc_val_scaled: np.ndarray
    X_desc_test_scaled: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    smiles_train: np.ndarray
    smiles_val: np.ndarray
    smiles_test: np.ndarray
    scaler: StandardScaler
    descriptor_names: list[str]


def load_scaffold_split_data(
    npz_path: str | Path,
    dataset_name: str | None = None,
) -> ScaffoldSplitDataset:
    """Load Phase 1 NPZ archive and standardize descriptors using train split only."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 feature archive not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        X_fp = archive["X_fp"].astype(np.float32)
        X_desc = archive["X_desc"].astype(np.float32)
        y = archive["y"]
        smiles = archive["smiles"]
        train_idx = archive["train_idx"]
        val_idx = archive["val_idx"]
        test_idx = archive["test_idx"]
        descriptor_names = [str(name) for name in archive["descriptor_names"]]

    scaler = StandardScaler()
    X_desc_train_scaled = scaler.fit_transform(X_desc[train_idx]).astype(np.float32)
    X_desc_val_scaled = scaler.transform(X_desc[val_idx]).astype(np.float32)
    X_desc_test_scaled = scaler.transform(X_desc[test_idx]).astype(np.float32)

    X_train = np.hstack([X_fp[train_idx], X_desc_train_scaled]).astype(np.float32)
    X_val = np.hstack([X_fp[val_idx], X_desc_val_scaled]).astype(np.float32)
    X_test = np.hstack([X_fp[test_idx], X_desc_test_scaled]).astype(np.float32)

    return ScaffoldSplitDataset(
        name=dataset_name or path.stem.replace("_features", ""),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        X_desc_train_scaled=X_desc_train_scaled,
        X_desc_val_scaled=X_desc_val_scaled,
        X_desc_test_scaled=X_desc_test_scaled,
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        smiles_train=smiles[train_idx],
        smiles_val=smiles[val_idx],
        smiles_test=smiles[test_idx],
        scaler=scaler,
        descriptor_names=descriptor_names,
    )


def evaluate_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute ROC-AUC, PR-AUC, F1, MCC, and Balanced Accuracy."""
    y_true_int = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob_arr >= threshold).astype(np.int64)

    return {
        "roc_auc": float(roc_auc_score(y_true_int, y_prob_arr)),
        "pr_auc": float(average_precision_score(y_true_int, y_prob_arr)),
        "f1": float(f1_score(y_true_int, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_int, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_int, y_pred)),
        "threshold": float(threshold),
    }


def evaluate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute RMSE, MAE, R2, and Spearman rank correlation."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    rho = float(spearmanr(y_true_arr, y_pred_arr).statistic)
    if np.isnan(rho):
        rho = 0.0

    return {
        "rmse": float(root_mean_squared_error(y_true_arr, y_pred_arr)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "spearman_r": rho,
    }


def select_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Select probability threshold on Validation split maximizing 0.5*(BalAcc + MCC)."""
    thresholds = grid if grid is not None else np.linspace(0.2, 0.8, 61)
    y_true_int = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    best_thr = 0.5
    best_score = -float("inf")
    for thr in thresholds:
        y_pred = (y_prob_arr >= thr).astype(np.int64)
        bal_acc = balanced_accuracy_score(y_true_int, y_pred)
        mcc = matthews_corrcoef(y_true_int, y_pred)
        score = 0.5 * (bal_acc + mcc)
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr


def _save_joblib_checkpoint(
    path: Path,
    model: Any,
    scaler: StandardScaler,
    descriptor_names: list[str],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist trained scikit-learn/XGBoost model bundle with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model,
        "scaler": scaler,
        "descriptor_names": descriptor_names,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if extra:
        payload.update(extra)
    joblib.dump(payload, path, compress=3)
    LOGGER.info("Saved baseline checkpoint -> %s", path)


def train_bbbp_baselines(
    data_dir: Path = Path("data/processed"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate Random Forest & XGBoost classifiers on BBBP scaffold split."""
    ds = load_scaffold_split_data(data_dir / "bbbp_features.npz", dataset_name="bbbp")

    n_pos = int((ds.y_train == 1).sum())
    n_neg = int((ds.y_train == 0).sum())
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    LOGGER.info(
        "BBBP Train class counts: %d BBB+ / %d BBB- (XGBoost scale_pos_weight=%.4f)",
        n_pos,
        n_neg,
        scale_pos_weight,
    )

    models = {
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
            checkpoint_dir / "rf_bbbp.joblib",
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=seed,
                n_jobs=-1,
            ),
            checkpoint_dir / "xgb_bbbp.joblib",
        ),
    }

    results: dict[str, Any] = {}
    for model_name, (clf, ckpt_path) in models.items():
        LOGGER.info("Training %s on BBBP (%d features)...", model_name, ds.X_train.shape[1])
        clf.fit(ds.X_train, ds.y_train)

        val_prob = clf.predict_proba(ds.X_val)[:, 1]
        opt_thr = select_optimal_threshold(ds.y_val, val_prob)
        val_metrics = evaluate_classification_metrics(ds.y_val, val_prob, threshold=opt_thr)

        test_prob = clf.predict_proba(ds.X_test)[:, 1]
        test_metrics = evaluate_classification_metrics(ds.y_test, test_prob, threshold=opt_thr)

        _save_joblib_checkpoint(
            path=ckpt_path,
            model=clf,
            scaler=ds.scaler,
            descriptor_names=ds.descriptor_names,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            extra={"model_name": model_name, "threshold": opt_thr},
        )
        LOGGER.info(
            "BBBP [%s] Val ROC-AUC=%.4f PR-AUC=%.4f | Test ROC-AUC=%.4f PR-AUC=%.4f MCC=%.4f",
            model_name,
            val_metrics["roc_auc"],
            val_metrics["pr_auc"],
            test_metrics["roc_auc"],
            test_metrics["pr_auc"],
            test_metrics["mcc"],
        )

        results[model_name] = {
            "val": val_metrics,
            "test": test_metrics,
            "y_test": ds.y_test.copy(),
            "test_prob": test_prob,
            "checkpoint": str(ckpt_path),
        }

    return results


def train_ache_baselines(
    data_dir: Path = Path("data/processed"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate Random Forest & XGBoost regressors on CHEMBL220 pIC50."""
    ds = load_scaffold_split_data(data_dir / "chembl220_features.npz", dataset_name="chembl220")

    models = {
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=-1,
            ),
            checkpoint_dir / "rf_ache.joblib",
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="rmse",
                random_state=seed,
                n_jobs=-1,
            ),
            checkpoint_dir / "xgb_ache.joblib",
        ),
    }

    results: dict[str, Any] = {}
    for model_name, (reg, ckpt_path) in models.items():
        LOGGER.info("Training %s on CHEMBL220 (%d features)...", model_name, ds.X_train.shape[1])
        reg.fit(ds.X_train, ds.y_train)

        val_pred = reg.predict(ds.X_val)
        val_metrics = evaluate_regression_metrics(ds.y_val, val_pred)

        test_pred = reg.predict(ds.X_test)
        test_metrics = evaluate_regression_metrics(ds.y_test, test_pred)

        _save_joblib_checkpoint(
            path=ckpt_path,
            model=reg,
            scaler=ds.scaler,
            descriptor_names=ds.descriptor_names,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            extra={"model_name": model_name},
        )
        LOGGER.info(
            "CHEMBL220 [%s] Val RMSE=%.4f R2=%.4f | Test RMSE=%.4f MAE=%.4f R2=%.4f Spearman=%.4f",
            model_name,
            val_metrics["rmse"],
            val_metrics["r2"],
            test_metrics["rmse"],
            test_metrics["mae"],
            test_metrics["r2"],
            test_metrics["spearman_r"],
        )

        results[model_name] = {
            "val": val_metrics,
            "test": test_metrics,
            "y_test": ds.y_test.copy(),
            "test_pred": test_pred,
            "checkpoint": str(ckpt_path),
        }

    return results
