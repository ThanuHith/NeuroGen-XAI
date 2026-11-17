"""Unified Phase 2 training, scientific benchmarking, and visualization runner.

Executes Random Forest, XGBoost, and GINEConv Graph Neural Network training on
both Phase 1 Bemis-Murcko scaffold-split tasks:
  - Task A: BBBP Blood-Brain Barrier Penetration (Classification)
  - Task B: CHEMBL220 Human Acetylcholinesterase pIC50 (Regression)

Generates:
  - `reports/benchmark_results.json`
  - `reports/benchmark_table.md`
  - `reports/figures/roc_pr_curves.png`
  - `reports/figures/pic50_parity_plot.png`
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.models.baseline_rf_xgb import train_ache_baselines, train_bbbp_baselines
from src.models.gnn_multitask import train_gnn_ache, train_gnn_bbbp


LOGGER = logging.getLogger("neurogen_xai.benchmark")


def _plot_roc_pr_curves(
    bbbp_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Save 2-panel side-by-side Test-set ROC and Precision-Recall curves (300 DPI)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=300)
    palette = {
        "RandomForest": "#1f77b4",
        "XGBoost": "#ff7f0e",
        "GINEConv_GNN": "#2ca02c",
    }

    # Left panel: ROC curves
    ax_roc = axes[0]
    for model_name, res in bbbp_results.items():
        y_true = res["y_test"]
        y_prob = res["test_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = res["test"]["roc_auc"]
        ax_roc.plot(
            fpr,
            tpr,
            lw=2.2,
            color=palette.get(model_name, None),
            label=f"{model_name} (ROC-AUC = {roc_auc:.3f})",
        )
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Chance")
    ax_roc.set_title("BBBP Test Set — ROC Curves (Scaffold Split)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontsize=10.5)
    ax_roc.set_ylabel("True Positive Rate", fontsize=10.5)
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.02)
    ax_roc.grid(alpha=0.25)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=9)

    # Right panel: Precision-Recall curves
    ax_pr = axes[1]
    first_res = next(iter(bbbp_results.values()))
    pos_baseline = float(np.mean(first_res["y_test"]))
    for model_name, res in bbbp_results.items():
        y_true = res["y_test"]
        y_prob = res["test_prob"]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = res["test"]["pr_auc"]
        ax_pr.plot(
            recall,
            precision,
            lw=2.2,
            color=palette.get(model_name, None),
            label=f"{model_name} (PR-AUC = {pr_auc:.3f})",
        )
    ax_pr.axhline(
        pos_baseline,
        color="black",
        linestyle="--",
        lw=1.2,
        alpha=0.6,
        label=f"No-Skill Baseline ({pos_baseline:.3f})",
    )
    ax_pr.set_title("BBBP Test Set — Precision-Recall Curves", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall", fontsize=10.5)
    ax_pr.set_ylabel("Precision", fontsize=10.5)
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.02)
    ax_pr.grid(alpha=0.25)
    ax_pr.legend(loc="lower left", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved ROC/PR figure -> %s", output_path)


def _plot_pic50_parity(
    ache_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Save 3-panel Experimental vs. Predicted pIC50 parity plots (300 DPI)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_order = list(ache_results.keys())
    fig, axes = plt.subplots(1, len(model_order), figsize=(5.2 * len(model_order), 5.0), dpi=300)
    if len(model_order) == 1:
        axes = [axes]

    palette = {
        "RandomForest": "#1f77b4",
        "XGBoost": "#ff7f0e",
        "GINEConv_GNN": "#2ca02c",
    }

    all_vals = np.concatenate(
        [np.concatenate([r["y_test"], r["test_pred"]]) for r in ache_results.values()]
    )
    vmin = float(np.floor(all_vals.min())) - 0.3
    vmax = float(np.ceil(all_vals.max())) + 0.3
    grid = np.linspace(vmin, vmax, 200)

    for ax, model_name in zip(axes, model_order):
        res = ache_results[model_name]
        y_true = res["y_test"]
        y_pred = res["test_pred"]
        m = res["test"]

        ax.scatter(
            y_true,
            y_pred,
            s=18,
            alpha=0.55,
            color=palette.get(model_name, "#1f77b4"),
            edgecolors="none",
        )
        ax.plot(grid, grid, "k-", lw=1.4, label="y = x")
        ax.plot(grid, grid + 1.0, "r--", lw=1.0, alpha=0.65, label="±1.0 log unit")
        ax.plot(grid, grid - 1.0, "r--", lw=1.0, alpha=0.65)

        metrics_text = (
            f"RMSE = {m['rmse']:.3f}\n"
            f"MAE = {m['mae']:.3f}\n"
            f"R² = {m['r2']:.3f}\n"
            f"Spearman ρ = {m['spearman_r']:.3f}"
        )
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        )
        ax.set_title(f"{model_name} (Test n={len(y_true)})", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Experimental pIC50", fontsize=10.5)
        ax.set_ylabel("Predicted pIC50", fontsize=10.5)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", frameon=True, fontsize=8.5)

    fig.suptitle(
        "CHEMBL220 (Human AChE) Test Set — Experimental vs. Predicted pIC50",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved pIC50 parity figure -> %s", output_path)


def _format_benchmark_markdown(summary: dict[str, Any]) -> str:
    """Render clean Markdown tables for Validation and Test benchmark metrics."""
    lines: list[str] = [
        "# NeuroGen-XAI Phase 2 Benchmark Results (Bemis-Murcko Scaffold Split)",
        "",
        "## Task A: Blood-Brain Barrier Penetration (BBBP — Binary Classification)",
        "",
        "| Split | Model | ROC-AUC | PR-AUC | F1-Score | MCC | Balanced Acc | Threshold |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for split in ("val", "test"):
        for model_name, model_data in summary["bbbp"].items():
            m = model_data[split]
            lines.append(
                f"| {split.upper()} | {model_name} | "
                f"{m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['f1']:.4f} | "
                f"{m['mcc']:.4f} | {m['balanced_accuracy']:.4f} | {m['threshold']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Task B: Human Acetylcholinesterase Potency (CHEMBL220 pIC50 — Regression)",
            "",
            "| Split | Model | RMSE | MAE | R² | Spearman ρ |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for split in ("val", "test"):
        for model_name, model_data in summary["chembl220"].items():
            m = model_data[split]
            lines.append(
                f"| {split.upper()} | {model_name} | "
                f"{m['rmse']:.4f} | {m['mae']:.4f} | {m['r2']:.4f} | {m['spearman_r']:.4f} |"
            )

    lines.append("")
    return "\n".join(lines)


def _print_test_summary(summary: dict[str, Any]) -> None:
    """Print formatted Test-set benchmark comparison tables to stdout."""
    print("\n" + "=" * 86)
    print("NEUROGEN-XAI PHASE 2 TEST-SET BENCHMARK SUMMARY (BEMIS-MURCKO SCAFFOLD SPLIT)")
    print("=" * 86)

    print("\n[Task A] BBBP Classification (Test Set)")
    print(
        f"{'Model':<16} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>9} "
        f"{'MCC':>9} {'BalAcc':>9} {'Thr':>7}"
    )
    print("-" * 74)
    for model_name, model_data in summary["bbbp"].items():
        m = model_data["test"]
        print(
            f"{model_name:<16} {m['roc_auc']:>9.4f} {m['pr_auc']:>9.4f} "
            f"{m['f1']:>9.4f} {m['mcc']:>9.4f} {m['balanced_accuracy']:>9.4f} {m['threshold']:>7.2f}"
        )

    print("\n[Task B] CHEMBL220 Human AChE pIC50 Regression (Test Set)")
    print(f"{'Model':<16} {'RMSE':>9} {'MAE':>9} {'R2':>9} {'Spearman':>11}")
    print("-" * 58)
    for model_name, model_data in summary["chembl220"].items():
        m = model_data["test"]
        print(
            f"{model_name:<16} {m['rmse']:>9.4f} {m['mae']:>9.4f} "
            f"{m['r2']:>9.4f} {m['spearman_r']:>11.4f}"
        )
    print("=" * 86 + "\n")


def run_benchmark(
    data_dir: Path = Path("data/processed"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    reports_dir: Path = Path("reports"),
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Train all Phase 2 models, generate diagnostic figures, and write reports."""
    bbbp_baseline = train_bbbp_baselines(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
    bbbp_gnn = train_gnn_bbbp(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
    )
    bbbp_all = {
        "RandomForest": bbbp_baseline["RandomForest"],
        "XGBoost": bbbp_baseline["XGBoost"],
        "GINEConv_GNN": bbbp_gnn,
    }

    ache_baseline = train_ache_baselines(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
    ache_gnn = train_gnn_ache(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
    )
    ache_all = {
        "RandomForest": ache_baseline["RandomForest"],
        "XGBoost": ache_baseline["XGBoost"],
        "GINEConv_GNN": ache_gnn,
    }

    figures_dir = reports_dir / "figures"
    _plot_roc_pr_curves(bbbp_all, figures_dir / "roc_pr_curves.png")
    _plot_pic50_parity(ache_all, figures_dir / "pic50_parity_plot.png")

    serializable_summary: dict[str, Any] = {
        "bbbp": {
            name: {
                "val": res["val"],
                "test": res["test"],
                "checkpoint": res["checkpoint"],
            }
            for name, res in bbbp_all.items()
        },
        "chembl220": {
            name: {
                "val": res["val"],
                "test": res["test"],
                "checkpoint": res["checkpoint"],
            }
            for name, res in ache_all.items()
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "benchmark_results.json"
    md_path = reports_dir / "benchmark_table.md"

    json_path.write_text(json.dumps(serializable_summary, indent=2), encoding="utf-8")
    md_path.write_text(_format_benchmark_markdown(serializable_summary), encoding="utf-8")
    LOGGER.info("Saved benchmark JSON -> %s", json_path)
    LOGGER.info("Saved benchmark Markdown -> %s", md_path)

    _print_test_summary(serializable_summary)
    return serializable_summary


def parse_args() -> argparse.Namespace:
    """Parse CLI options for Phase 2 training and benchmarking."""
    parser = argparse.ArgumentParser(
        description="NeuroGen-XAI Phase 2: Baseline + GINEConv GNN Training & Benchmarking"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    run_benchmark(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
