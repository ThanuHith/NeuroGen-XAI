# NeuroGen-XAI

**NeuroGen-XAI** is an open-source, research-grade computational drug discovery platform combining **Explainable AI (XAI)** and **De Novo Molecular Design** to discover and optimize Blood-Brain Barrier (BBB) penetrant Central Nervous System (CNS) therapeutics.

---

## Architecture Overview

```text
neurogen-xai/
├── data/
│   ├── raw/                        # Raw MoleculeNet BBBP & ChEMBL220 IC50 downloads
│   └── processed/                  # Curated CSVs, NPZ features (2048-bit Morgan + 8 descriptors), stats JSON
├── src/
│   ├── data_pipeline/
│   │   └── prepare_datasets.py     # Phase 1: Download, RDKit standardization, featurization, Bemis-Murcko split
│   ├── models/
│   │   ├── baseline_rf_xgb.py      # Phase 2A: Random Forest & XGBoost baselines on Morgan FP + descriptors
│   │   └── gnn_multitask.py        # Phase 2B: PyTorch Geometric Multi-Task GNN (BBB+ & AChE pIC50)
│   ├── generative/
│   │   └── mol_generator.py        # Phase 3: SELFIES/RL-based constrained de novo molecule generator
│   └── xai/
│       └── explainer.py            # Phase 4: TreeSHAP/KernelSHAP & RDKit 2D atom-level attribution heatmaps
├── app/
│   └── main.py                     # Streamlit interactive UI for inference, XAI heatmaps, & de novo design
├── environment.yml                 # Reproducible Conda/Mamba environment specification
├── Dockerfile                      # Containerized runtime for pipeline execution and Streamlit app
└── README.md
```

---

## Development Phases

- **Phase 1 — Data Curation & Scaffold-Split Pipeline (`src/data_pipeline/prepare_datasets.py`)** *(Implemented)*:
  - Fetches MoleculeNet **BBBP** (`p_np` BBB permeability binary labels) and **Human Acetylcholinesterase (`CHEMBL220`)** $\text{IC}_{50}$ bioactivity records (converted to $\text{pIC}_{50} = 9.0 - \log_{10}(\text{IC}_{50\text{ [nM]}})$).
  - Performs RDKit chemical standardization: invalid SMILES filtering, salt/fragment stripping (`LargestFragmentChooser`), charge neutralization (`Uncharger`), isomeric canonicalization, and duplicate/conflict resolution.
  - Computes **8 CNS-relevant physicochemical descriptors** (`MW`, `LogP`, `TPSA`, `HBD`, `HBA`, `RotBonds`, `QED`, `SA_Score`) and **2048-bit Morgan Fingerprints (radius 2)**.
  - Performs deterministic **80% Train / 10% Validation / 10% Test Bemis-Murcko Scaffold Splitting** with strict zero-leakage assertions.
- **Phase 2 — Multi-Task Predictive Modeling (`src/models/`)** *(Scaffolded)*:
  - Baseline Random Forest & XGBoost models (`baseline_rf_xgb.py`) trained on Morgan fingerprints + physicochemical descriptors.
  - Multi-Task PyTorch Geometric GNN (`gnn_multitask.py`) using GIN/GATv2 message passing for joint BBB+ classification and AChE $\text{pIC}_{50}$ regression.
- **Phase 3 — Constrained De Novo Generation (`src/generative/mol_generator.py`)** *(Scaffolded)*:
  - SELFIES-based RL / evolutionary molecular generator optimizing a multi-objective CNS reward: predicted $\text{pIC}_{50}$, BBB+ probability, Synthetic Accessibility (`SA_Score`), `QED`, and PAINS / hERG liability filters.
- **Phase 4 — Explainable AI & Web App (`src/xai/explainer.py`, `app/main.py`)** *(Scaffolded)*:
  - SHAP attribution for fingerprint/descriptor models and atom-level importance heatmaps projected onto 2D RDKit molecular depictions inside a Streamlit web interface.

---

## Quickstart

### 1. Environment Setup (Conda / Mamba)

```bash
mamba env create -f environment.yml
conda activate neurogen-xai
```

### 2. Run Phase 1 Data Curation & Scaffold Splitting

```bash
python -m src.data_pipeline.prepare_datasets
```

Optional CLI arguments:
- `--data-dir`: Base data directory (default: `data`)
- `--train-frac`, `--val-frac`, `--test-frac`: Scaffold split ratios (default: `0.8`, `0.1`, `0.1`)
- `--seed`: Random seed (default: `42`)
- `--max-chembl-records`: Optional cap on fetched ChEMBL220 records (default: fetch all available records)
- `--force-download`: Re-download raw datasets even if cached in `data/raw/`

### 3. Docker Usage

```bash
# Build image
docker build -t neurogen-xai .

# Run Phase 1 pipeline inside container
docker run --rm -v "$(pwd)/data:/app/data" neurogen-xai \
    python -m src.data_pipeline.prepare_datasets

# Launch Streamlit web interface
docker run --rm -p 8501:8501 neurogen-xai
```
