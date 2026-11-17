"""Phase 4B: Interactive Streamlit Research Platform for NeuroGen-XAI.

Three-tab scientific dashboard:
  1. Single Molecule Analyzer & Dual-XAI Atom Attribution
  2. Constrained De Novo CNS Lead Generator (SELFIES + RL/Evolutionary)
  3. Model Benchmarks & Bemis-Murcko Scaffold-Split Dataset Transparency
"""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.generative.mol_generator import (
    DEFAULT_CNS_SEEDS,
    ConstrainedSELFIESGenerator,
    GenerationConstraints,
    StructuralAlertFilter,
)
from src.xai.explainer import NeuroGenExplainer


DATA_STATS_PATH = ROOT_DIR / "data/processed/dataset_statistics.json"
BENCHMARK_JSON_PATH = ROOT_DIR / "reports/benchmark_results.json"
GENERATED_CSV_PATH = ROOT_DIR / "reports/generated_cns_candidates.csv"
GENERATED_GRID_PATH = ROOT_DIR / "reports/figures/top_generated_molecules.png"
ROC_PR_FIG_PATH = ROOT_DIR / "reports/figures/roc_pr_curves.png"
PARITY_FIG_PATH = ROOT_DIR / "reports/figures/pic50_parity_plot.png"

REFERENCE_DRUGS: dict[str, dict[str, str]] = {
    "Donepezil (FDA AChE Inhibitor, BBB+)": {
        "smiles": DEFAULT_CNS_SEEDS["Donepezil"],
        "note": "Selective reversible AChE inhibitor for Alzheimer's disease (CNS active).",
    },
    "Tacrine (Historic AChE Inhibitor, BBB+)": {
        "smiles": DEFAULT_CNS_SEEDS["Tacrine"],
        "note": "First FDA-approved AChE inhibitor with high BBB penetration.",
    },
    "Rivastigmine (FDA AChE Inhibitor, BBB+)": {
        "smiles": DEFAULT_CNS_SEEDS["Rivastigmine"],
        "note": "Pseudo-irreversible carbamate AChE/BuChE inhibitor for dementia.",
    },
    "Galantamine (FDA AChE Inhibitor, BBB+)": {
        "smiles": DEFAULT_CNS_SEEDS["Galantamine"],
        "note": "Alkaloid AChE inhibitor and allosteric nicotinic receptor modulator.",
    },
    "Caffeine (CNS Stimulant, BBB+)": {
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "note": "Reference small-molecule CNS penetrant negative for AChE potency.",
    },
    "Atenolol (Peripheral Beta-Blocker, BBB-)": {
        "smiles": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
        "note": "Negative BBB permeability control (high TPSA / low CNS penetration).",
    },
}


def _apply_custom_theme() -> None:
    """Inject crisp White, Clinical Blue, and Jet Black clinical biotech CSS styling."""
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #FFFFFF !important;
            color: #0A0F1C !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #F4F8FF !important;
            border-right: 1px solid #C8DCFF !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0A0F1C !important;
            font-weight: 700 !important;
        }
        p, span, li, label,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] p,
        label[data-testid="stWidgetLabel"] p {
            color: #111827 !important;
        }
        .neuro-banner {
            padding: 1.25rem 1.5rem;
            border-radius: 10px;
            background-color: #F4F8FF;
            border: 1px solid #C8DCFF;
            border-left: 5px solid #0052CC;
            border-top: 3px solid #0052CC;
            margin-bottom: 1.25rem;
        }
        .neuro-banner h1 {
            margin: 0;
            font-size: 1.85rem;
            color: #0A0F1C !important;
            letter-spacing: -0.02em;
        }
        .neuro-banner p {
            margin: 0.4rem 0 0 0;
            color: #1E293B !important;
            font-size: 0.96rem;
            font-weight: 500;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.65rem 0 1.1rem 0;
        }
        .pill {
            display: inline-block;
            padding: 0.30rem 0.78rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            border: 1px solid transparent;
        }
        .pill-pass {
            background-color: #ECFDF5;
            color: #065F46 !important;
            border-color: #6EE7B7;
        }
        .pill-fail {
            background-color: #FEF2F2;
            color: #991B1B !important;
            border-color: #FCA5A5;
        }
        .pill-info {
            background-color: #EFF6FF;
            color: #0052CC !important;
            border-color: #BFDBFE;
        }
        div[data-testid="stMetric"] {
            background-color: #F4F8FF !important;
            border: 1px solid #C8DCFF !important;
            border-top: 3px solid #0052CC !important;
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] p {
            color: #0A0F1C !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #0052CC !important;
            font-weight: 700 !important;
        }
        div[data-testid="stExpander"] {
            background-color: #F4F8FF !important;
            border: 1px solid #C8DCFF !important;
            border-radius: 10px;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary p {
            color: #0A0F1C !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"] {
            color: #1E293B !important;
            font-weight: 600 !important;
            background-color: transparent !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #0052CC !important;
            font-weight: 700 !important;
            border-bottom-color: #0052CC !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        input {
            background-color: #FFFFFF !important;
            color: #0A0F1C !important;
            border-color: #93C5FD !important;
        }
        div.stButton > button,
        div.stDownloadButton > button {
            background-color: #0052CC !important;
            color: #FFFFFF !important;
            border: 1px solid #0052CC !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }
        div.stButton > button p,
        div.stDownloadButton > button p,
        div.stButton > button span,
        div.stDownloadButton > button span {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            background-color: #003E99 !important;
            border-color: #003E99 !important;
            color: #FFFFFF !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #C8DCFF !important;
            border-radius: 8px;
        }
        div[data-testid="stDataFrame"] th,
        div[data-testid="stDataFrame"] td {
            color: #0A0F1C !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading NeuroGen-XAI Ensemble (RF + XGBoost + GINEConv)...")
def load_explainer() -> NeuroGenExplainer:
    """Cache the 6-model predictive and XAI explainer across Streamlit reruns."""
    return NeuroGenExplainer(checkpoint_dir=ROOT_DIR / "models/checkpoints")


@st.cache_resource(show_spinner="Loading Phase 1 reference corpus...")
def load_alert_filter() -> StructuralAlertFilter:
    """Cache PAINS, Brenk, and hERG structural alert filter."""
    return StructuralAlertFilter()


def _plot_descriptor_shap(
    desc_shap_bbb: dict[str, float],
    desc_shap_ache: dict[str, float],
) -> plt.Figure:
    """Render side-by-side horizontal bar charts of physicochemical SHAP impacts."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")

    for ax, shap_map, title in (
        (axes[0], desc_shap_bbb, "Physicochemical SHAP — BBB+ (Log-Odds)"),
        (axes[1], desc_shap_ache, "Physicochemical SHAP — AChE pIC50"),
    ):
        ax.set_facecolor("#FFFFFF")
        items = sorted(shap_map.items(), key=lambda kv: abs(kv[1]))
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        colors = ["#0052CC" if v >= 0 else "#DC2626" for v in vals]

        ax.barh(names, vals, color=colors, height=0.62)
        ax.axvline(0.0, color="#0A0F1C", linewidth=1.0, linestyle="--")
        ax.set_title(title, color="#0A0F1C", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#0A0F1C", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
        ax.grid(axis="x", alpha=0.35, color="#CBD5E1")

    fig.tight_layout()
    return fig


def _plot_cns_admet_radar(desc: dict[str, float]) -> plt.Figure:
    """Render a normalized CNS MPO / drug-likeness radar chart."""
    axes_specs = [
        ("MW\n(<=450)", max(0.0, 1.0 - max(0.0, desc["MW"] - 250.0) / 350.0)),
        ("LogP\n(2-4)", max(0.0, 1.0 - abs(desc["LogP"] - 3.0) / 4.0)),
        ("TPSA\n(<=90)", max(0.0, 1.0 - desc["TPSA"] / 140.0)),
        ("HBD\n(<=3)", max(0.0, 1.0 - desc["HBD"] / 5.0)),
        ("QED\n(0-1)", float(np.clip(desc["QED"], 0.0, 1.0))),
        ("Synth.\n(10-SA)/9", float(np.clip((10.0 - desc["SA_Score"]) / 9.0, 0.0, 1.0))),
    ]
    labels = [s[0] for s in axes_specs]
    values = [s[1] for s in axes_specs]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(4.4, 3.8), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#F4F8FF")
    ax.plot(angles_closed, values_closed, color="#0052CC", linewidth=2.2)
    ax.fill(angles_closed, values_closed, color="#0052CC", alpha=0.22)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color="#0A0F1C", fontsize=8.0, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="#1E293B", fontsize=7.5)
    ax.grid(color="#CBD5E1", alpha=0.8)
    ax.spines["polar"].set_color("#93C5FD")
    ax.set_title("CNS Drug-Likeness Profile", color="#0A0F1C", fontsize=10, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def _render_mol_png(smiles: str, width: int = 340, height: int = 230) -> Image.Image | None:
    """Render a clean 2D RDKit molecule image for display cards."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().clearBackground = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def _render_tab_analyzer(explainer: NeuroGenExplainer, alert_filter: StructuralAlertFilter) -> None:
    """Tab 1: Single Molecule Analyzer & Dual-XAI Atom Attribution."""
    st.subheader("Single Molecule Analyzer & Dual-XAI Attribution")

    col_sel, col_smi = st.columns([1.2, 2.0])
    with col_sel:
        preset_names = list(REFERENCE_DRUGS.keys()) + ["Custom SMILES"]
        selected_preset = st.selectbox("Reference Molecule Preset", preset_names, index=0)

    default_smiles = (
        REFERENCE_DRUGS[selected_preset]["smiles"]
        if selected_preset in REFERENCE_DRUGS
        else DEFAULT_CNS_SEEDS["Donepezil"]
    )
    with col_smi:
        smiles_input = st.text_input("Canonical / Isomeric SMILES", value=default_smiles)

    if selected_preset in REFERENCE_DRUGS:
        st.caption(REFERENCE_DRUGS[selected_preset]["note"])

    if not smiles_input.strip():
        st.warning("Enter a valid SMILES string to analyze.")
        return

    try:
        with st.spinner("Running ensemble inference & Dual-XAI atom attribution..."):
            attr = explainer.explain_molecule(smiles_input.strip(), width=540, height=400)
    except Exception as exc:
        st.error(f"Could not parse or analyze SMILES: {exc}")
        return

    pred = attr.prediction
    desc = pred["descriptors"]
    mol = Chem.MolFromSmiles(attr.canonical_smiles)
    assert mol is not None
    alerts = alert_filter.inspect(mol, logp=desc["LogP"], sa_score=desc["SA_Score"])

    # KPI Metric Row
    m1, m2, m3, m4 = st.columns(4)
    bbb_p = pred["bbb_prob_ensemble"]
    m1.metric(
        "Ensemble P(BBB+)",
        f"{bbb_p:.3f}",
        delta="BBB Penetrant" if bbb_p >= 0.5 else "Low BBB Penetration",
        delta_color="normal" if bbb_p >= 0.5 else "inverse",
    )
    m2.metric(
        "Human AChE pIC50",
        f"{pred['pic50_ensemble']:.2f}",
        delta=f"IC50 ≈ {pred['ic50_nm']:.1f} nM",
    )
    m3.metric("QED Drug-Likeness", f"{desc['QED']:.3f}", delta=f"MW {desc['MW']:.1f} Da")
    m4.metric(
        "SA Score (1-10)",
        f"{desc['SA_Score']:.2f}",
        delta="Synthesizable (<=4.5)" if desc["SA_Score"] <= 4.5 else "Complex (>4.5)",
        delta_color="normal" if desc["SA_Score"] <= 4.5 else "inverse",
    )

    # Compliance & Model Breakdown Badges
    lip_cls = "pill-pass" if pred["lipinski_pass"] else "pill-fail"
    pains_cls = "pill-pass" if alerts["passes_pains"] else "pill-fail"
    brenk_cls = "pill-pass" if alerts["passes_brenk"] else "pill-fail"
    herg_cls = "pill-pass" if alerts["passes_herg"] else "pill-fail"
    st.markdown(
        f"""
        <div class="badge-row">
            <span class="pill {lip_cls}">Lipinski Ro5: {'PASS' if pred['lipinski_pass'] else 'FAIL'} ({pred['lipinski_violations']} violations)</span>
            <span class="pill {pains_cls}">PAINS: {'CLEAN' if alerts['passes_pains'] else 'ALERT'}</span>
            <span class="pill {brenk_cls}">Brenk: {'CLEAN' if alerts['passes_brenk'] else 'ALERT'}</span>
            <span class="pill {herg_cls}">hERG Heuristic: {'LOW RISK' if alerts['passes_herg'] else 'RISK ALERT'}</span>
            <span class="pill pill-info">BBBP RF/XGB/GNN: {pred['bbb_prob_rf']:.2f} / {pred['bbb_prob_xgb']:.2f} / {pred['bbb_prob_gnn']:.2f}</span>
            <span class="pill pill-info">pIC50 RF/XGB/GNN: {pred['pic50_rf']:.2f} / {pred['pic50_xgb']:.2f} / {pred['pic50_gnn']:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Side-by-Side 2D Atom-Level Attribution Heatmaps
    st.markdown("#### Atom-Level Dual XAI Attribution Maps (TreeSHAP + GINEConv Saliency)")
    st.caption(
        "Green contours highlight atoms/substructures driving BBB+ permeability or higher AChE potency; "
        "red contours highlight limiting or unfavorable substructures."
    )
    x_col1, x_col2 = st.columns(2)
    with x_col1:
        st.image(
            attr.png_bbb,
            caption=f"BBB+ Permeability Attribution | P(BBB+) = {pred['bbb_prob_ensemble']:.3f}",
            use_container_width=True,
        )
    with x_col2:
        st.image(
            attr.png_ache,
            caption=f"Human AChE pIC50 Attribution | pIC50 = {pred['pic50_ensemble']:.2f}",
            use_container_width=True,
        )

    # Descriptor SHAP & ADMET Radar
    st.markdown("#### Physicochemical SHAP Contributions & CNS Property Profile")
    c_shap, c_radar = st.columns([1.8, 1.0])
    with c_shap:
        fig_shap = _plot_descriptor_shap(attr.descriptor_shap_bbb, attr.descriptor_shap_ache)
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)
    with c_radar:
        fig_radar = _plot_cns_admet_radar(desc)
        st.pyplot(fig_radar, use_container_width=True)
        plt.close(fig_radar)

    desc_df = pd.DataFrame([desc]).round(3)
    st.dataframe(desc_df, use_container_width=True, hide_index=True)


def _render_tab_generator(explainer: NeuroGenExplainer) -> None:
    """Tab 2: Constrained De Novo CNS Lead Generator."""
    st.subheader("Constrained SELFIES / RL De Novo CNS Lead Generator")
    st.caption(
        "Generates novel AChE-potent, BBB-penetrant candidates via policy-weighted SELFIES mutations "
        "and crossover, strictly filtered against PAINS, Brenk, hERG risk, and SA_Score limits."
    )

    with st.expander("Multi-Objective Reward & Medicinal Chemistry Constraints", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        w_pic50 = c1.slider("pIC50 Weight", 0.0, 1.0, 0.40, 0.05)
        w_bbb = c2.slider("P(BBB+) Weight", 0.0, 1.0, 0.35, 0.05)
        w_qed = c3.slider("QED Weight", 0.0, 1.0, 0.15, 0.05)
        w_sa = c4.slider("SA Penalty Weight", 0.0, 0.50, 0.10, 0.05)

        s1, s2, s3, s4 = st.columns([1.3, 1.0, 1.0, 1.0])
        seed_options = ["All 4 CNS Leads"] + list(DEFAULT_CNS_SEEDS.keys())
        seed_choice = s1.selectbox("Seed Scaffolds", seed_options, index=0)
        max_sa = s2.slider("Max SA Score", 2.5, 6.0, 4.5, 0.1)
        generations = s3.slider("RL/Evo Generations", 4, 24, 10, 2)
        pop_size = s4.slider("Candidates / Gen", 16, 64, 32, 8)

    run_live = st.button("Generate Novel CNS Candidates Live", type="primary")

    if run_live:
        constraints = GenerationConstraints(
            w_pic50=w_pic50,
            w_bbb=w_bbb,
            w_qed=w_qed,
            w_sa=w_sa,
            max_sa_score=max_sa,
        )
        selected_seeds = (
            DEFAULT_CNS_SEEDS
            if seed_choice == "All 4 CNS Leads"
            else {seed_choice: DEFAULT_CNS_SEEDS[seed_choice]}
        )
        with st.spinner(
            f"Running {generations} generations ({pop_size} candidates/gen) with PAINS/Brenk/hERG/SA filters..."
        ):
            gen = ConstrainedSELFIESGenerator(
                explainer=explainer,
                constraints=constraints,
                processed_dir=ROOT_DIR / "data/processed",
                seed=42,
            )
            df_live = gen.generate(
                seed_smiles=selected_seeds,
                generations=generations,
                population_size=pop_size,
                top_k=10,
            )
        st.session_state["generated_df"] = df_live
        st.session_state["generated_source"] = "Live Run"

    if "generated_df" in st.session_state:
        df_show: pd.DataFrame = st.session_state["generated_df"]
        source_label = st.session_state.get("generated_source", "Live Run")
    elif GENERATED_CSV_PATH.exists():
        df_show = pd.read_csv(GENERATED_CSV_PATH)
        source_label = "Pre-Generated CLI Frontier"
    else:
        df_show = pd.DataFrame()
        source_label = "None"

    if df_show.empty:
        st.info(
            "No generated candidates found yet. Click **Generate Novel CNS Candidates Live** "
            "or run `py -m src.generative.mol_generator`."
        )
        return

    st.markdown(f"#### Top {len(df_show)} Novel CNS Candidates ({source_label})")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Top Reward", f"{df_show['reward'].max():.4f}")
    k2.metric("Best Predicted pIC50", f"{df_show['pic50_ensemble'].max():.2f}")
    k3.metric("Mean P(BBB+)", f"{df_show['bbb_prob_ensemble'].mean():.3f}")
    k4.metric("Mean Novelty (1-MaxSim)", f"{df_show['novelty_score'].mean():.3f}")

    display_cols = [
        "rank",
        "canonical_smiles",
        "reward",
        "pic50_ensemble",
        "ic50_nm",
        "bbb_prob_ensemble",
        "qed",
        "sa_score",
        "logp",
        "max_train_tanimoto",
        "novelty_score",
        "lipinski_pass",
        "passes_pains",
        "passes_brenk",
        "passes_herg",
        "seed_origin",
    ]
    available_cols = [c for c in display_cols if c in df_show.columns]
    st.dataframe(df_show[available_cols], use_container_width=True, hide_index=True)

    csv_bytes = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Candidates CSV",
        data=csv_bytes,
        file_name="neurogen_xai_generated_cns_candidates.csv",
        mime="text/csv",
    )

    # Render inline 2D previews for the top 5 leads (plus static grid if from CLI)
    st.markdown("#### 2D Chemical Structure Previews (Top 5 Leads)")
    preview_cols = st.columns(min(5, len(df_show)))
    for idx, col in enumerate(preview_cols):
        row = df_show.iloc[idx]
        img = _render_mol_png(str(row["canonical_smiles"]))
        if img is not None:
            col.image(
                img,
                caption=(
                    f"#{int(row['rank'])} | pIC50 {row['pic50_ensemble']:.2f} | "
                    f"BBB+ {row['bbb_prob_ensemble']:.2f} | SA {row['sa_score']:.2f}"
                ),
                use_container_width=True,
            )

    if source_label == "Pre-Generated CLI Frontier" and GENERATED_GRID_PATH.exists():
        with st.expander("Full 10-Molecule Publication Grid Figure"):
            st.image(str(GENERATED_GRID_PATH), use_container_width=True)


def _render_tab_benchmarks() -> None:
    """Tab 3: Model Benchmarks & Bemis-Murcko Scaffold-Split Dataset Transparency."""
    st.subheader("Phase 1 Scaffold-Split Statistics & Phase 2 Model Benchmarks")

    if DATA_STATS_PATH.exists():
        stats: dict[str, Any] = json.loads(DATA_STATS_PATH.read_text(encoding="utf-8"))
        st.markdown("#### Phase 1: RDKit Curation & Bemis-Murcko Scaffold Split (Zero Overlap)")
        c_bbbp, c_chembl = st.columns(2)

        for col, key, title in (
            (c_bbbp, "bbbp", "MoleculeNet BBBP (BBB Permeability)"),
            (c_chembl, "chembl220", "ChEMBL220 (Human Acetylcholinesterase IC50)"),
        ):
            ds = stats[key]
            clean = ds["cleaning_report"]
            splits = ds["splits"]
            with col:
                st.markdown(f"**{title}**")
                st.caption(
                    f"Raw: {clean['raw_records']} -> Final Unique: {clean['final_unique_molecules']} | "
                    f"Scaffolds: {splits['overall']['n_unique_scaffolds']} | "
                    f"Train/Val/Test Overlap: {ds['scaffold_overlap_check']['train_val_overlap']}/"
                    f"{ds['scaffold_overlap_check']['train_test_overlap']}/"
                    f"{ds['scaffold_overlap_check']['val_test_overlap']}"
                )
                split_rows = [
                    {
                        "Split": sname.upper(),
                        "Molecules": splits[sname]["n_molecules"],
                        "Unique Scaffolds": splits[sname]["n_unique_scaffolds"],
                        **(
                            {"BBB+ Rate": splits[sname]["positive_rate"]}
                            if "positive_rate" in splits[sname]
                            else {"pIC50 Mean": splits[sname]["target_mean"]}
                        ),
                    }
                    for sname in ("train", "val", "test")
                ]
                st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Dataset statistics JSON not found.")

    if BENCHMARK_JSON_PATH.exists():
        bench: dict[str, Any] = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
        st.markdown("#### Phase 2: Model Benchmarking Across Scaffold Splits")

        bbbp_rows = []
        for split in ("val", "test"):
            for mname, mdata in bench["bbbp"].items():
                bbbp_rows.append({"Split": split.upper(), "Model": mname, **mdata[split]})

        ache_rows = []
        for split in ("val", "test"):
            for mname, mdata in bench["chembl220"].items():
                ache_rows.append({"Split": split.upper(), "Model": mname, **mdata[split]})

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**Task A: BBBP Classification**")
            st.dataframe(pd.DataFrame(bbbp_rows).round(4), use_container_width=True, hide_index=True)
        with b2:
            st.markdown("**Task B: CHEMBL220 pIC50 Regression**")
            st.dataframe(pd.DataFrame(ache_rows).round(4), use_container_width=True, hide_index=True)
    else:
        st.info("Benchmark JSON not found.")

    st.markdown("#### Diagnostic Performance Curves")
    f1, f2 = st.columns(2)
    with f1:
        if ROC_PR_FIG_PATH.exists():
            st.image(str(ROC_PR_FIG_PATH), caption="BBBP Test Set ROC & PR Curves", use_container_width=True)
    with f2:
        if PARITY_FIG_PATH.exists():
            st.image(
                str(PARITY_FIG_PATH),
                caption="CHEMBL220 Test Set Experimental vs. Predicted pIC50",
                use_container_width=True,
            )


def render_app() -> None:
    """Main Streamlit UI entrypoint."""
    st.set_page_config(
        page_title="NeuroGen-XAI | CNS Drug Discovery",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_custom_theme()

    st.markdown(
        """
        <div class="neuro-banner">
            <h1>NeuroGen-XAI: Explainable CNS Drug Discovery & De Novo Design</h1>
            <p>
                Multi-Model Ensemble (Random Forest + XGBoost + PyG GINEConv) for Blood-Brain Barrier
                Permeability, Human Acetylcholinesterase (CHEMBL220) Potency, Dual-XAI Attribution,
                and Constrained SELFIES Molecule Generation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    explainer = load_explainer()
    alert_filter = load_alert_filter()

    tab1, tab2, tab3 = st.tabs(
        [
            "Single Molecule Analyzer & XAI Attribution",
            "De Novo CNS Lead Generator",
            "Model Benchmarks & Dataset Transparency",
        ]
    )

    with tab1:
        _render_tab_analyzer(explainer, alert_filter)
    with tab2:
        _render_tab_generator(explainer)
    with tab3:
        _render_tab_benchmarks()


if __name__ == "__main__":
    render_app()
