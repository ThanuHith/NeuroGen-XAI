"""Generate the comprehensive NeuroGen-XAI Story & Technical PDF Guide.

Reads verified Phase 1 statistics, Phase 2 benchmark results, Phase 3 XAI
visualizations, and Phase 4 generated CNS candidates to produce a clean,
publication-grade PDF in clinical White, Blue (#0052CC), and Jet Black (#0A0F1C).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PDF = ROOT_DIR / "reports" / "NeuroGen_XAI_Complete_Project_Guide.pdf"

BLUE = colors.HexColor("#0052CC")
DARK_BLUE = colors.HexColor("#003E99")
ICE_BLUE = colors.HexColor("#F4F8FF")
BORDER_BLUE = colors.HexColor("#C8DCFF")
JET_BLACK = colors.HexColor("#0A0F1C")
CHARCOAL = colors.HexColor("#1E293B")
LIGHT_GRAY = colors.HexColor("#E2E8F0")


def _scaled_image(path: Path, max_width: float, max_height: float = 4.2 * inch) -> Image:
    """Load an image preserving aspect ratio within max_width x max_height."""
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = min(max_width / w, max_height / h)
    return Image(str(path), width=w * ratio, height=h * ratio)


def _header_footer(canvas, doc) -> None:
    """Draw clinical header rule and footer page number on every page."""
    canvas.saveState()
    page_w, page_h = letter
    if doc.page > 1:
        canvas.setStrokeColor(BORDER_BLUE)
        canvas.setLineWidth(0.8)
        canvas.line(0.65 * inch, page_h - 0.48 * inch, page_w - 0.65 * inch, page_h - 0.48 * inch)
        canvas.setFont("Helvetica-Bold", 8.5)
        canvas.setFillColor(BLUE)
        canvas.drawString(0.65 * inch, page_h - 0.40 * inch, "NeuroGen-XAI")
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(CHARCOAL)
        canvas.drawRightString(
            page_w - 0.65 * inch,
            page_h - 0.40 * inch,
            "Explainable AI & De Novo CNS Drug Discovery Guide",
        )

    canvas.setStrokeColor(BORDER_BLUE)
    canvas.setLineWidth(0.8)
    canvas.line(0.65 * inch, 0.50 * inch, page_w - 0.65 * inch, 0.50 * inch)
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(CHARCOAL)
    canvas.drawString(0.65 * inch, 0.34 * inch, "NeuroGen-XAI Open-Source Platform Documentation")
    canvas.drawRightString(page_w - 0.65 * inch, 0.34 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _styled_table(
    data: list[list[str]],
    col_widths: list[float],
    font_size: float = 8.5,
) -> Table:
    """Build a crisp clinical blue/white striped table."""
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ICE_BLUE]),
                ("TEXTCOLOR", (0, 1), (-1, -1), JET_BLACK),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER_BLUE),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def _callout_box(title: str, body: str, styles: dict[str, ParagraphStyle], width: float) -> Table:
    """Create a light ice-blue callout box with a thick clinical blue left accent."""
    content = [[Paragraph(f"<b>{title}</b><br/>{body}", styles["Callout"])]]
    tbl = Table(content, colWidths=[width])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ICE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.75, BORDER_BLUE),
                ("LINEBEFORE", (0, 0), (0, -1), 4.0, BLUE),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return tbl


def build_pdf() -> Path:
    """Build the complete NeuroGen-XAI PDF report."""
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    usable_w = letter[0] - 1.3 * inch

    base_styles = getSampleStyleSheet()
    styles = {
        "CoverTitle": ParagraphStyle(
            "CoverTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            textColor=JET_BLACK,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "CoverSub": ParagraphStyle(
            "CoverSub",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=BLUE,
            spaceAfter=14,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15.5,
            leading=20,
            textColor=BLUE,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=JET_BLACK,
            spaceBefore=9,
            spaceAfter=4,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.8,
            leading=14.2,
            textColor=CHARCOAL,
            spaceAfter=6,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.8,
            leftIndent=14,
            bulletIndent=4,
            textColor=CHARCOAL,
            spaceAfter=4,
        ),
        "Code": ParagraphStyle(
            "Code",
            parent=base_styles["Code"],
            fontName="Courier",
            fontSize=8.3,
            leading=11.5,
            backColor=ICE_BLUE,
            borderPadding=6,
            textColor=JET_BLACK,
            spaceAfter=7,
        ),
        "Callout": ParagraphStyle(
            "Callout",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=13.5,
            textColor=JET_BLACK,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base_styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11.5,
            alignment=TA_CENTER,
            textColor=CHARCOAL,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.1,
            leading=11.0,
            textColor=JET_BLACK,
            alignment=TA_LEFT,
        ),
        "TableCellBold": ParagraphStyle(
            "TableCellBold",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.1,
            leading=11.0,
            textColor=JET_BLACK,
            alignment=TA_LEFT,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.3,
            leading=11.0,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
    }

    # Load verified project JSON / CSV data
    stats = json.loads((ROOT_DIR / "data/processed/dataset_statistics.json").read_text(encoding="utf-8"))
    bench = json.loads((ROOT_DIR / "reports/benchmark_results.json").read_text(encoding="utf-8"))
    gen_df = pd.read_csv(ROOT_DIR / "reports/generated_cns_candidates.csv")

    story = []

    # --- COVER / CHAPTER 1 ---
    banner = Table(
        [
            [Paragraph("NEUROGEN-XAI | COMPLETE PROJECT GUIDE", ParagraphStyle("Eyebrow", fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=BLUE))],
            [Paragraph("Explainable AI &amp; De Novo Molecular Design for Brain-Penetrant CNS Drugs", ParagraphStyle("BannerTitle", fontName="Helvetica-Bold", fontSize=18.5, leading=23, textColor=JET_BLACK))],
            [Paragraph("From Raw SMILES Molecules to Leakage-Free Benchmarks, Dual-XAI Atom Heatmaps, and Synthesizable Novel Therapeutics", ParagraphStyle("BannerSub", fontName="Helvetica", fontSize=9.8, leading=13.5, textColor=CHARCOAL))],
        ],
        colWidths=[usable_w],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ICE_BLUE),
                ("BOX", (0, 0), (-1, -1), 1.0, BORDER_BLUE),
                ("LINEABOVE", (0, 0), (-1, 0), 4.0, BLUE),
                ("TOPPADDING", (0, 0), (-1, 0), 12),
                ("TOPPADDING", (0, 1), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -2), 2),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    story.append(banner)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. The Story: Why We Built NeuroGen-XAI", styles["H1"]))
    story.append(
        Paragraph(
            "Imagine trying to send a repair crew into a high-security bank vault. Designing medicines for "
            "Alzheimer's disease and other Central Nervous System (CNS) disorders faces two connected hurdles:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "• <b>The Therapeutic Target (Human Acetylcholinesterase — CHEMBL220):</b> In Alzheimer's disease, "
            "the memory neurotransmitter acetylcholine breaks down too quickly. Inhibiting the enzyme "
            "Acetylcholinesterase (AChE) preserves acetylcholine signaling and relieves cognitive decline.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>The Fortress Wall (The Blood-Brain Barrier — BBB):</b> The brain is protected by a tightly "
            "sealed cellular wall that blocks over 98% of small-molecule drugs. A compound can bind AChE tightly "
            "in a lab dish yet fail completely in patients if it cannot cross the BBB.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "Traditional AI models create two more bottlenecks: (1) <b>Black-Box Opacity</b> — predicting a score "
            "without showing chemists <i>which atoms</i> help or hurt permeability and potency; and (2) "
            "<b>Unsynthesizable or Toxic AI Hallucinations</b> — generating molecules that break chemical rules, "
            "trigger Pan-Assay Interference (PAINS) alerts, or carry hERG cardiac toxicity risk.",
            styles["Body"],
        )
    )
    story.append(
        _callout_box(
            "How NeuroGen-XAI Solves All Four Problems in One End-to-End Pipeline",
            "<b>Phase 1:</b> Curates standardized, leakage-free datasets using RDKit and Bemis-Murcko scaffold splitting.<br/>"
            "<b>Phase 2:</b> Benchmarks Random Forest, XGBoost, and a PyTorch Geometric GINEConv Graph Neural Network.<br/>"
            "<b>Phase 3:</b> Explains every prediction down to individual atoms via Dual-XAI (TreeSHAP + GNN Saliency).<br/>"
            "<b>Phase 4:</b> Invents novel, synthesizable, PAINS/Brenk/hERG-clean CNS candidates using constrained SELFIES.",
            styles,
            usable_w,
        )
    )

    # --- CHAPTER 2: INPUT -> OUTPUT ---
    story.append(Paragraph("2. How the ML System Works: Input to Output Walkthrough", styles["H1"]))
    story.append(
        Paragraph(
            "Computers cannot read 2D chemical drawings directly. In NeuroGen-XAI, every molecule enters as a "
            "<b>SMILES</b> text string (for example, Donepezil: "
            "<font name='Courier'>COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2</font>) and flows through five steps:",
            styles["Body"],
        )
    )
    flow_raw = [
        ("Stage", "What Happens Inside the System", "Concrete Example (Donepezil)"),
        (
            "1. Standardize\n(RDKit)",
            "Parse SMILES, strip salt counterions (LargestFragmentChooser), neutralize charges (Uncharger), and canonicalize.",
            "Validates 28 heavy atoms, neutralizes charge, and produces a unique canonical SMILES.",
        ),
        (
            "2. Dual\nFeaturization",
            "Branch A (Trees): 2048-bit Morgan FP (radius 2) + 8 scaled descriptors (2056 dims). Branch B (GNN): 46-dim atom nodes + 12-dim bidirectional bond edges + 8 descriptors.",
            "MW=379.50, LogP=4.36, TPSA=38.77, HBD=0, HBA=4, RotBonds=6, QED=0.747, SA=2.677.",
        ),
        (
            "3. 6-Model\nEnsemble",
            "Averages predictions of Random Forest + XGBoost + GINEConv GNN on BBB+ probability and AChE pIC50.",
            "P(BBB+) = 0.9589 (RF=0.89, XGB=1.00, GNN=0.99); pIC50 = 6.973 (IC50 ≈ 106.30 nM); Lipinski PASS.",
        ),
        (
            "4. Dual-XAI\nAttribution",
            "Maps TreeSHAP + RF bit ablation via RDKit bitInfo and fuses with GNN Input*Gradient + Atom Occlusion.",
            "Highlights lipophilic indanone, piperidine, and benzyl atoms in green and polar =O in red for BBB+.",
        ),
        (
            "5. SELFIES\nDe Novo Design",
            "Mutates & crosses over CNS seeds in guaranteed-valid SELFIES grammar; enforces PAINS, Brenk, hERG, and SA filters.",
            "Discovers novel analog O=C1COCCC2CN(Cc3ccccc3)CCC2C1 (pIC50 7.42, P(BBB+) 0.951, SA 2.97).",
        ),
    ]
    flow_data = [
        [
            Paragraph(flow_raw[0][0], styles["TableHeader"]),
            Paragraph(flow_raw[0][1], styles["TableHeader"]),
            Paragraph(flow_raw[0][2], styles["TableHeader"]),
        ]
    ]
    for stage, desc, ex in flow_raw[1:]:
        flow_data.append(
            [
                Paragraph(stage.replace("\n", "<br/>"), styles["TableCellBold"]),
                Paragraph(desc, styles["TableCell"]),
                Paragraph(ex, styles["TableCell"]),
            ]
        )
    flow_tbl = Table(flow_data, colWidths=[1.10 * inch, 3.15 * inch, usable_w - 4.25 * inch], repeatRows=1)
    flow_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ICE_BLUE]),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER_BLUE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(flow_tbl)
    story.append(Spacer(1, 8))

    # --- CHAPTER 3: PHASE 1 ---
    story.append(Paragraph("3. Phase 1: Data Curation & Bemis-Murcko Scaffold Splitting", styles["H1"]))
    story.append(
        Paragraph(
            "<b>Module:</b> <font name='Courier'>src/data_pipeline/prepare_datasets.py</font><br/>"
            "Public drug datasets contain salt fragments (.HCl, .Na), charged states, malformed SMILES, and noisy "
            "replicate assays. Phase 1 fetches and cleans two complementary benchmarks:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "• <b>MoleculeNet BBBP:</b> 2,050 raw records downloaded from DeepChem S3. After dropping 11 invalid SMILES, "
            "collapsing 62 duplicates, and removing 22 compounds with conflicting 0/1 labels, <b>1,955 unique molecules</b> "
            "(1,494 BBB+ / 461 BBB-, 76.42% positive; 1,015 unique scaffolds) remain.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>ChEMBL220 (Human Acetylcholinesterase):</b> 7,827 exact IC50 (nM) records fetched via paginated ChEMBL REST API. "
            "Converted to logarithmic potency via <b>pIC50 = 9.0 - log10(IC50_nM)</b> (so 1 nM = 9.0, 100 nM = 7.0). "
            "After dropping 1 invalid SMILES, aggregating 1,485 replicates by median pIC50, and removing 269 noisy entries "
            "(replicate std &gt; 1.0), <b>6,071 unique molecules</b> (2,547 unique scaffolds; mean pIC50 5.956 ± 1.492) remain.",
            styles["Bullet"],
        )
    )
    story.append(Paragraph("Why Bemis-Murcko Scaffold Splitting Prevents Cheating", styles["H2"]))
    story.append(
        Paragraph(
            "If molecules from the same chemical series are split randomly across Train and Test, models memorize "
            "scaffolds instead of learning medicinal chemistry. We extract each molecule's core ring-and-linker skeleton "
            "(<b>Bemis-Murcko Scaffold</b>), sort scaffold buckets deterministically by (-size, scaffold_smiles), and assign "
            "entire scaffold families into <b>80% Train / 10% Validation / 10% Test</b> with automated assertions verifying "
            "<b>0 overlapping scaffolds</b> between any two splits.",
            styles["Body"],
        )
    )

    p1_rows = [
        ["Dataset", "Raw", "Final Unique", "Scaffolds", "Train (80%)", "Val (10%)", "Test (10%)", "Overlap"],
        [
            "BBBP (BBB+/-)",
            str(stats["bbbp"]["cleaning_report"]["raw_records"]),
            str(stats["bbbp"]["cleaning_report"]["final_unique_molecules"]),
            str(stats["bbbp"]["splits"]["overall"]["n_unique_scaffolds"]),
            f"{stats['bbbp']['splits']['train']['n_molecules']} ({stats['bbbp']['splits']['train']['n_unique_scaffolds']} scaf)",
            f"{stats['bbbp']['splits']['val']['n_molecules']} ({stats['bbbp']['splits']['val']['n_unique_scaffolds']} scaf)",
            f"{stats['bbbp']['splits']['test']['n_molecules']} ({stats['bbbp']['splits']['test']['n_unique_scaffolds']} scaf)",
            "0",
        ],
        [
            "CHEMBL220 (pIC50)",
            str(stats["chembl220"]["cleaning_report"]["raw_records"]),
            str(stats["chembl220"]["cleaning_report"]["final_unique_molecules"]),
            str(stats["chembl220"]["splits"]["overall"]["n_unique_scaffolds"]),
            f"{stats['chembl220']['splits']['train']['n_molecules']} ({stats['chembl220']['splits']['train']['n_unique_scaffolds']} scaf)",
            f"{stats['chembl220']['splits']['val']['n_molecules']} ({stats['chembl220']['splits']['val']['n_unique_scaffolds']} scaf)",
            f"{stats['chembl220']['splits']['test']['n_molecules']} ({stats['chembl220']['splits']['test']['n_unique_scaffolds']} scaf)",
            "0",
        ],
    ]
    story.append(_styled_table(p1_rows, [1.25 * inch, 0.55 * inch, 0.8 * inch, 0.65 * inch, 1.15 * inch, 1.05 * inch, 1.05 * inch, usable_w - 6.50 * inch], font_size=8.0))
    story.append(PageBreak())

    # --- CHAPTER 4: PHASE 2 ---
    story.append(Paragraph("4. Phase 2: ML Algorithms, Metrics & Benchmarks", styles["H1"]))
    story.append(
        Paragraph(
            "<b>Modules:</b> <font name='Courier'>src/models/baseline_rf_xgb.py</font>, "
            "<font name='Courier'>gnn_multitask.py</font>, <font name='Courier'>train_and_benchmark.py</font><br/>"
            "A <font name='Courier'>StandardScaler</font> is fit strictly on the 8 physicochemical descriptors in the "
            "Train split before concatenating with 2048-bit Morgan fingerprints (2,056 total features).",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "• <b>Random Forest:</b> 500 bagged trees (<font name='Courier'>class_weight='balanced'</font> on BBBP) "
            "providing high-variance stability on circular substructure bits.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>XGBoost:</b> Gradient boosted decision trees (<font name='Courier'>max_depth=6, lr=0.05</font>) with "
            "<font name='Courier'>scale_pos_weight=0.3448</font> on BBBP to balance 1,163 BBB+ vs. 401 BBB- training molecules.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>4-Layer GINEConv Graph Neural Network (PyTorch Geometric):</b> Converts each molecule into a graph of "
            "<b>46-dim atom vectors</b> (element, degree, formal charge, hybridization, aromaticity, H-count, chirality, ring) "
            "and <b>12-dim bidirectional bond vectors</b> (bond order, conjugation, ring, stereochemistry). Four GINEConv layers "
            "with residual connections and BatchNorm1d pass messages between bonded atoms, pool via mean+max readout (256-dim), "
            "fuse with an encoded 32-dim descriptor branch, and train with weighted BCEWithLogitsLoss (BBBP) or HuberLoss (AChE) "
            "plus CosineAnnealingLR and early stopping (patience 12).",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>Metrics Guide:</b> For BBBP, <b>ROC-AUC</b> measures ranking separation, <b>PR-AUC</b> evaluates precision-recall "
            "under class skew, <b>MCC</b> (&isin; [-1, +1]) requires strong performance on <i>both</i> BBB+ and BBB- classes, and "
            "<b>Balanced Accuracy</b> averages sensitivity and specificity using thresholds tuned on Validation to maximize "
            "0.5*(BalAcc + MCC). For AChE regression, <b>RMSE</b> &amp; <b>MAE</b> measure error in pIC50 log units, <b>R2</b> "
            "measures variance explained, and <b>Spearman rho</b> measures hit-ranking fidelity.",
            styles["Bullet"],
        )
    )

    bbbp_tbl = [["Split", "Model", "ROC-AUC", "PR-AUC", "F1-Score", "MCC", "Balanced Acc", "Val Thr"]]
    for sp in ("val", "test"):
        for mname, mdata in bench["bbbp"].items():
            m = mdata[sp]
            bbbp_tbl.append([
                sp.upper(), mname, f"{m['roc_auc']:.4f}", f"{m['pr_auc']:.4f}",
                f"{m['f1']:.4f}", f"{m['mcc']:.4f}", f"{m['balanced_accuracy']:.4f}", f"{m['threshold']:.2f}",
            ])
    story.append(Paragraph("Task A: BBBP Permeability Classification Results", styles["H2"]))
    story.append(_styled_table(bbbp_tbl, [0.6 * inch, 1.2 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.8 * inch, 1.0 * inch, usable_w - 6.15 * inch], font_size=8.2))
    story.append(Spacer(1, 6))

    ache_tbl = [["Split", "Model", "RMSE (pIC50)", "MAE (pIC50)", "R2", "Spearman rho"]]
    for sp in ("val", "test"):
        for mname, mdata in bench["chembl220"].items():
            m = mdata[sp]
            ache_tbl.append([
                sp.upper(), mname, f"{m['rmse']:.4f}", f"{m['mae']:.4f}",
                f"{m['r2']:.4f}", f"{m['spearman_r']:.4f}",
            ])
    story.append(Paragraph("Task B: CHEMBL220 Human AChE pIC50 Regression Results", styles["H2"]))
    story.append(_styled_table(ache_tbl, [0.75 * inch, 1.45 * inch, 1.2 * inch, 1.2 * inch, 1.1 * inch, usable_w - 5.7 * inch], font_size=8.2))
    story.append(Spacer(1, 8))

    story.append(_scaled_image(ROOT_DIR / "reports/figures/roc_pr_curves.png", usable_w, 2.75 * inch))
    story.append(Paragraph("Figure 1: BBBP Test-Set ROC and Precision-Recall Curves (Zero-Overlap Scaffold Split).", styles["Caption"]))
    story.append(_scaled_image(ROOT_DIR / "reports/figures/pic50_parity_plot.png", usable_w, 2.55 * inch))
    story.append(Paragraph("Figure 2: CHEMBL220 Test-Set Experimental vs. Predicted pIC50 Parity Plots.", styles["Caption"]))
    story.append(PageBreak())

    # --- CHAPTER 5: PHASE 3 ---
    story.append(Paragraph("5. Phase 3: Dual Explainable AI (XAI) Engine", styles["H1"]))
    story.append(
        Paragraph(
            "<b>Module:</b> <font name='Courier'>src/xai/explainer.py</font><br/>"
            "<font name='Courier'>NeuroGenExplainer</font> averages the 3 BBBP models and 3 AChE models into a unified "
            "6-model ensemble, then builds atom-level heatmaps by fusing two independent explanation methods:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "1. <b>TreeSHAP + RF Bit Ablation Mapped via RDKit bitInfo:</b> Computes exact Shapley contributions "
            "(<font name='Courier'>shap.TreeExplainer</font> on XGBoost) and active-bit perturbation deltas on Random Forest "
            "across all 2,056 features. Using RDKit's <font name='Courier'>bitInfo</font> and "
            "<font name='Courier'>FindAtomEnvironmentOfRadiusN</font>, each active Morgan bit's SHAP credit is distributed "
            "back onto the exact atoms belonging to that circular chemical fragment.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "2. <b>GINEConv Input*Gradient Saliency + Batched Atom Occlusion:</b> Computes gradient sensitivity of the GNN output "
            "with respect to each atom's 46-dim feature vector, averaged with the prediction drop when each atom's features are "
            "individually masked out in a single batched forward pass.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "3. <b>Fusion &amp; Visualization:</b> Both pathways are normalized into [-1, +1], averaged (50/50), and rendered "
            "onto 2D RDKit drawings with <font name='Courier'>SimilarityMaps</font> (<font name='Courier'>RdYlGn</font>): "
            "<b>Green contours</b> mark atoms increasing BBB+ permeability or AChE potency; <b>Red contours</b> mark limiting moieties.",
            styles["Bullet"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(_scaled_image(ROOT_DIR / "reports/figures/xai_donepezil_attribution.png", usable_w, 3.3 * inch))
    story.append(
        Paragraph(
            "Figure 3: Donepezil Dual-XAI Attribution. Left: BBB+ Permeability (P=0.959; green lipophilic piperidine/benzyl &amp; "
            "indanone rings vs. red polar =O). Right: AChE Potency (pIC50=6.97, IC50 ≈ 106.3 nM; green dimethoxyindanone core).",
            styles["Caption"],
        )
    )

    # --- CHAPTER 6: PHASE 4 ---
    story.append(Paragraph("6. Phase 4: Constrained SELFIES Generator & Web Platform", styles["H1"]))
    story.append(
        Paragraph(
            "<b>Modules:</b> <font name='Courier'>src/generative/mol_generator.py</font> &amp; "
            "<font name='Courier'>app/main.py</font><br/>"
            "Mutating raw SMILES text frequently breaks parentheses or ring closures, creating invalid strings. "
            "<b>SELFIES</b> encodes molecules so that <i>every</i> token replacement, insertion, deletion, or 1-point crossover "
            "decodes into a 100% valid chemical graph.",
            styles["Body"],
        )
    )
    story.append(
        _callout_box(
            "Multi-Objective Reward &amp; Medicinal Chemistry Gatekeepers",
            "<b>Reward = 0.40 * norm(pIC50) + 0.35 * P(BBB+) + 0.15 * QED - 0.10 * norm(SA_Score)</b><br/>"
            "Every candidate must pass: (1) <b>PAINS</b> filter, (2) <b>Brenk</b> toxicophore filter, "
            "(3) <b>hERG</b> heuristic (rejects lipophilic basic amines with LogP &gt; 5.0 or 2–4 carbon aromatic tethers at LogP &gt; 4.0), "
            "(4) <b>SA_Score &lt;= 4.5</b> (lab synthesizability), (5) <b>150 &lt;= MW &lt;= 550</b>, and (6) <b>Novelty check</b> "
            "(Morgan Tanimoto &lt; 0.999 against all 7,998 unique Phase 1 training molecules).",
            styles,
            usable_w,
        )
    )
    story.append(Spacer(1, 8))

    gen_tbl = [["Rank", "Reward", "pIC50", "IC50 (nM)", "P(BBB+)", "QED", "SA", "MaxSim", "Canonical SMILES"]]
    for _, row in gen_df.iterrows():
        gen_tbl.append([
            str(int(row["rank"])),
            f"{row['reward']:.4f}",
            f"{row['pic50_ensemble']:.2f}",
            f"{row['ic50_nm']:.1f}",
            f"{row['bbb_prob_ensemble']:.3f}",
            f"{row['qed']:.2f}",
            f"{row['sa_score']:.2f}",
            f"{row['max_train_tanimoto']:.2f}",
            str(row["canonical_smiles"]),
        ])
    story.append(Paragraph("Top 10 Generated Novel CNS Candidates (16 Generations x 48 Candidates)", styles["H2"]))
    story.append(_styled_table(gen_tbl, [0.4 * inch, 0.55 * inch, 0.5 * inch, 0.65 * inch, 0.6 * inch, 0.45 * inch, 0.45 * inch, 0.55 * inch, usable_w - 4.15 * inch], font_size=7.4))
    story.append(Spacer(1, 8))
    story.append(_scaled_image(ROOT_DIR / "reports/figures/top_generated_molecules.png", usable_w, 3.6 * inch))
    story.append(Paragraph("Figure 4: Top 10 Generated Novel CNS Drug Candidates (All PAINS, Brenk, hERG &amp; SA Filtered).", styles["Caption"]))

    # --- CHAPTER 7: STREAMLIT, CODEBASE MAP & COMMANDS ---
    story.append(PageBreak())
    story.append(Paragraph("7. Interactive Streamlit Platform, Codebase Map &amp; CLI Commands", styles["H1"]))
    story.append(
        Paragraph(
            "Configured with a high-contrast <b>White, Clinical Blue (#0052CC), and Jet Black (#0A0F1C)</b> theme in "
            "<font name='Courier'>.streamlit/config.toml</font> and <font name='Courier'>app/main.py</font>, the platform offers "
            "3 interactive tabs: (1) <b>Single Molecule Analyzer &amp; XAI Attribution</b>, (2) <b>De Novo CNS Lead Generator</b> "
            "(live reward/SA/seed sliders + CSV export), and (3) <b>Model Benchmarks &amp; Dataset Transparency</b>.",
            styles["Body"],
        )
    )

    code_map_raw = [
        ("Project File", "Phase & Responsibility"),
        ("src/data_pipeline/prepare_datasets.py", "Phase 1: Download BBBP & ChEMBL220, RDKit cleaning, Morgan FP + 8 descriptors, Bemis-Murcko 80/10/10 split."),
        ("src/models/baseline_rf_xgb.py", "Phase 2A: Train Random Forest & XGBoost on 2,056 features; tune Val threshold; save .joblib checkpoints."),
        ("src/models/gnn_multitask.py", "Phase 2B: 46-dim atom / 12-dim bond PyG graph featurizer & 4-layer GINEConv training/saving."),
        ("src/models/train_and_benchmark.py", "Phase 2 Runner: Benchmarks all 6 models and exports JSON, Markdown, ROC/PR & parity plots."),
        ("src/xai/explainer.py", "Phase 3: 6-model ensemble predictor + TreeSHAP/GNN Dual-XAI atom attribution heatmaps."),
        ("src/generative/mol_generator.py", "Phase 4A: Constrained SELFIES RL/evolutionary generator + PAINS/Brenk/hERG/SA/Tanimoto filters."),
        ("app/main.py & .streamlit/config.toml", "Phase 4B: Clinical White/Blue/Black Streamlit research web platform."),
    ]
    code_map_data = [
        [Paragraph(code_map_raw[0][0], styles["TableHeader"]), Paragraph(code_map_raw[0][1], styles["TableHeader"])]
    ] + [
        [Paragraph(f"<font name='Courier'>{f}</font>", styles["TableCellBold"]), Paragraph(r, styles["TableCell"])]
        for f, r in code_map_raw[1:]
    ]
    code_tbl = Table(code_map_data, colWidths=[2.45 * inch, usable_w - 2.45 * inch], repeatRows=1)
    code_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ICE_BLUE]),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER_BLUE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(code_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("CLI Quick Reference", styles["H2"]))
    story.append(
        Paragraph(
            "py -m src.data_pipeline.prepare_datasets   # Phase 1: Curate &amp; scaffold-split<br/>"
            "py -m src.models.train_and_benchmark       # Phase 2: Train RF, XGBoost, GINEConv<br/>"
            "py -m src.xai.explainer                    # Phase 3: Donepezil Dual-XAI figure<br/>"
            "py -m src.generative.mol_generator         # Phase 4A: Generate Top 10 CNS leads<br/>"
            "py -m streamlit run app/main.py            # Phase 4B: Launch interactive dashboard",
            styles["Code"],
        )
    )

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="NeuroGen-XAI Complete Project Story & Technical Guide",
        author="NeuroGen-XAI Platform",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return OUTPUT_PDF


if __name__ == "__main__":
    pdf_path = build_pdf()
    print(f"PDF_BUILT_OK: {pdf_path} ({pdf_path.stat().st_size} bytes)")
