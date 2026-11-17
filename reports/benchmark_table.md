# NeuroGen-XAI Phase 2 Benchmark Results (Bemis-Murcko Scaffold Split)

## Task A: Blood-Brain Barrier Penetration (BBBP — Binary Classification)

| Split | Model | ROC-AUC | PR-AUC | F1-Score | MCC | Balanced Acc | Threshold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| VAL | RandomForest | 0.8768 | 0.9641 | 0.8767 | 0.5416 | 0.8155 | 0.68 |
| VAL | XGBoost | 0.8843 | 0.9690 | 0.8805 | 0.5497 | 0.8187 | 0.77 |
| VAL | GINEConv_GNN | 0.8349 | 0.9504 | 0.8931 | 0.4222 | 0.7022 | 0.55 |
| TEST | RandomForest | 0.8584 | 0.9804 | 0.9249 | 0.3602 | 0.6870 | 0.68 |
| TEST | XGBoost | 0.7649 | 0.9641 | 0.9148 | 0.1668 | 0.5763 | 0.77 |
| TEST | GINEConv_GNN | 0.8229 | 0.9726 | 0.9363 | 0.2309 | 0.5765 | 0.55 |

## Task B: Human Acetylcholinesterase Potency (CHEMBL220 pIC50 — Regression)

| Split | Model | RMSE | MAE | R² | Spearman ρ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| VAL | RandomForest | 1.0892 | 0.7945 | 0.4387 | 0.6863 |
| VAL | XGBoost | 1.0380 | 0.7589 | 0.4902 | 0.7303 |
| VAL | GINEConv_GNN | 1.1176 | 0.8251 | 0.4089 | 0.6783 |
| TEST | RandomForest | 1.0792 | 0.8123 | 0.5015 | 0.6930 |
| TEST | XGBoost | 1.0725 | 0.8093 | 0.5076 | 0.6881 |
| TEST | GINEConv_GNN | 1.1804 | 0.8762 | 0.4036 | 0.6452 |
