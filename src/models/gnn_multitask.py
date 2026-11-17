"""PyTorch Geometric GINEConv Graph Neural Networks for BBBP and CHEMBL220 pIC50.

Phase 2B implementation:
  - Converts RDKit molecules from canonical SMILES into PyG `Data` objects with
    46-dim atom node features, 12-dim bidirectional bond edge features, and
    standardized 8-dim physicochemical descriptors.
  - Implements `NeuroGINEModel` (4-layer GINEConv with residual connections,
    BatchNorm1d, global mean+max pooling, and physicochemical fusion head).
  - Trains `GNN_BBBP` (BCEWithLogitsLoss + positive class weighting, early stopping
    on Val ROC-AUC/PR-AUC) and `GNN_AChE` (HuberLoss, early stopping on Val RMSE).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from src.models.baseline_rf_xgb import (
    ScaffoldSplitDataset,
    evaluate_classification_metrics,
    evaluate_regression_metrics,
    load_scaffold_split_data,
    select_optimal_threshold,
)


LOGGER = logging.getLogger("neurogen_xai.models.gnn")

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "I", "B", "Na", "K", "Li", "Ca"]
DEGREES = [0, 1, 2, 3, 4]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
TOTAL_HS = [0, 1, 2, 3]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]

NODE_FEAT_DIM = 46
EDGE_FEAT_DIM = 12
DESC_FEAT_DIM = 8


def _one_hot_with_unknown(value: Any, choices: list[Any]) -> list[float]:
    """Encode value as one-hot vector with an appended 'other/unknown' bucket."""
    vec = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    vec[idx] = 1.0
    return vec


def _one_hot_strict(value: Any, choices: list[Any]) -> list[float]:
    """Encode value as fixed-length one-hot vector (defaults to index 0 if unmatched)."""
    vec = [0.0] * len(choices)
    if value in choices:
        vec[choices.index(value)] = 1.0
    else:
        vec[0] = 1.0
    return vec


def _atom_features(atom: Chem.Atom, is_potential_chiral: bool) -> list[float]:
    """Build 46-dimensional RDKit atom feature vector."""
    symbol_oh = _one_hot_with_unknown(atom.GetSymbol(), ATOM_SYMBOLS)  # 16
    degree_oh = _one_hot_with_unknown(atom.GetTotalDegree(), DEGREES)  # 6
    charge_oh = _one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGES)  # 6
    hyb_oh = _one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS)  # 6
    aromatic = [1.0 if atom.GetIsAromatic() else 0.0]  # 1
    hydrogens_oh = _one_hot_with_unknown(atom.GetTotalNumHs(includeNeighbors=True), TOTAL_HS)  # 5
    chiral_oh = _one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS)  # 4
    chiral_possible = [1.0 if is_potential_chiral else 0.0]  # 1
    in_ring = [1.0 if atom.IsInRing() else 0.0]  # 1

    feats = (
        symbol_oh
        + degree_oh
        + charge_oh
        + hyb_oh
        + aromatic
        + hydrogens_oh
        + chiral_oh
        + chiral_possible
        + in_ring
    )
    return feats


def _bond_features(bond: Chem.Bond) -> list[float]:
    """Build 12-dimensional RDKit bond feature vector."""
    btype_oh = _one_hot_strict(bond.GetBondType(), BOND_TYPES)  # 4
    conjugated = [1.0 if bond.GetIsConjugated() else 0.0]  # 1
    in_ring = [1.0 if bond.IsInRing() else 0.0]  # 1
    stereo_oh = _one_hot_strict(bond.GetStereo(), BOND_STEREOS)  # 6
    return btype_oh + conjugated + in_ring + stereo_oh


def smiles_to_pyg_data(
    smiles: str,
    descriptors: np.ndarray | None = None,
    y: float | int | None = None,
) -> Data:
    """Convert a SMILES string into a PyTorch Geometric Data molecular graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES passed to smiles_to_pyg_data: {smiles}")

    chiral_centers = {idx for idx, _ in Chem.FindMolChiralCenters(mol, includeUnassigned=True)}
    node_feats = [
        _atom_features(atom, atom.GetIdx() in chiral_centers)
        for atom in mol.GetAtoms()
    ]
    x = torch.tensor(node_feats, dtype=torch.float32)

    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)

    desc_arr = (
        np.zeros((1, DESC_FEAT_DIM), dtype=np.float32)
        if descriptors is None
        else np.asarray(descriptors, dtype=np.float32).reshape(1, DESC_FEAT_DIM)
    )
    u = torch.from_numpy(desc_arr.copy())

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, smiles=smiles)
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    return data


@dataclass
class GINEConfig:
    """Hyperparameters for NeuroGINEModel."""

    node_in_dim: int = NODE_FEAT_DIM
    edge_in_dim: int = EDGE_FEAT_DIM
    desc_in_dim: int = DESC_FEAT_DIM
    hidden_dim: int = 128
    desc_hidden_dim: int = 32
    num_layers: int = 4
    dropout: float = 0.2


class NeuroGINEModel(nn.Module):
    """4-layer GINEConv Graph Neural Network fused with physicochemical descriptors."""

    def __init__(self, config: GINEConfig | None = None) -> None:
        super().__init__()
        self.config = config or GINEConfig()
        cfg = self.config

        self.node_encoder = nn.Sequential(
            nn.Linear(cfg.node_in_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            mlp = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.BatchNorm1d(cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=True, edge_dim=cfg.edge_in_dim))
            self.norms.append(nn.BatchNorm1d(cfg.hidden_dim))

        self.desc_encoder = nn.Sequential(
            nn.Linear(cfg.desc_in_dim, cfg.desc_hidden_dim),
            nn.LayerNorm(cfg.desc_hidden_dim),
            nn.SiLU(),
        )

        head_in_dim = cfg.hidden_dim * 2 + cfg.desc_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning scalar prediction/logit of shape (batch_size,)."""
        h = self.node_encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            h_next = conv(h, edge_index, edge_attr)
            h_next = norm(h_next)
            h = h + F.dropout(F.silu(h_next), p=self.config.dropout, training=self.training)

        graph_repr = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        desc_repr = self.desc_encoder(u)
        fused = torch.cat([graph_repr, desc_repr], dim=-1)
        return self.head(fused).squeeze(-1)


def _set_seed(seed: int) -> None:
    """Set deterministic seeds across Python/NumPy/PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_graph_splits(ds: ScaffoldSplitDataset) -> tuple[list[Data], list[Data], list[Data]]:
    """Construct PyG graph lists for Train, Val, and Test splits."""
    train_graphs = [
        smiles_to_pyg_data(smi, desc, y)
        for smi, desc, y in zip(ds.smiles_train, ds.X_desc_train_scaled, ds.y_train)
    ]
    val_graphs = [
        smiles_to_pyg_data(smi, desc, y)
        for smi, desc, y in zip(ds.smiles_val, ds.X_desc_val_scaled, ds.y_val)
    ]
    test_graphs = [
        smiles_to_pyg_data(smi, desc, y)
        for smi, desc, y in zip(ds.smiles_test, ds.X_desc_test_scaled, ds.y_test)
    ]
    return train_graphs, val_graphs, test_graphs


@torch.no_grad()
def _predict_gnn(
    model: NeuroGINEModel,
    loader: DataLoader,
    device: torch.device,
    task: Literal["classification", "regression"],
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched inference and return (y_true, y_pred_or_prob)."""
    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
        if task == "classification":
            out = torch.sigmoid(out)
        ys.append(batch.y.view(-1).cpu().numpy())
        preds.append(out.view(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def _train_gnn_core(
    ds: ScaffoldSplitDataset,
    task: Literal["classification", "regression"],
    checkpoint_path: Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Shared GINEConv training, early stopping, evaluation, and checkpoint saving."""
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training GINEConv (%s / %s) on device=%s", ds.name.upper(), task, device)

    train_graphs, val_graphs, test_graphs = _build_graph_splits(ds)
    drop_last_train = len(train_graphs) % batch_size == 1
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    cfg = GINEConfig()
    model = NeuroGINEModel(cfg).to(device)

    if task == "classification":
        n_pos = int((ds.y_train == 1).sum())
        n_neg = int((ds.y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_score = -float("inf")
    else:
        criterion = nn.HuberLoss(delta=1.0)
        best_score = float("inf")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += float(loss.item()) * batch.num_graphs
            n_samples += batch.num_graphs

        scheduler.step()
        train_loss = total_loss / max(n_samples, 1)

        y_val, val_out = _predict_gnn(model, val_loader, device, task=task)
        if task == "classification":
            val_metrics = evaluate_classification_metrics(y_val, val_out, threshold=0.5)
            monitor_score = 0.5 * (val_metrics["roc_auc"] + val_metrics["pr_auc"])
            improved = monitor_score > best_score
        else:
            val_metrics = evaluate_regression_metrics(y_val, val_out)
            monitor_score = val_metrics["rmse"]
            improved = monitor_score < best_score

        if improved:
            best_score = monitor_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 10 == 0 or epochs_no_improve == patience:
            if task == "classification":
                LOGGER.info(
                    "  Epoch %02d/%02d | loss=%.4f | Val ROC-AUC=%.4f PR-AUC=%.4f",
                    epoch,
                    epochs,
                    train_loss,
                    val_metrics["roc_auc"],
                    val_metrics["pr_auc"],
                )
            else:
                LOGGER.info(
                    "  Epoch %02d/%02d | loss=%.4f | Val RMSE=%.4f Spearman=%.4f",
                    epoch,
                    epochs,
                    train_loss,
                    val_metrics["rmse"],
                    val_metrics["spearman_r"],
                )

        if epochs_no_improve >= patience:
            LOGGER.info("Early stopping triggered at epoch %d (best epoch=%d).", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    y_val, val_out = _predict_gnn(model, val_loader, device, task=task)
    y_test, test_out = _predict_gnn(model, test_loader, device, task=task)

    extra_ckpt: dict[str, Any] = {}
    if task == "classification":
        opt_thr = select_optimal_threshold(y_val, val_out)
        val_metrics = evaluate_classification_metrics(y_val, val_out, threshold=opt_thr)
        test_metrics = evaluate_classification_metrics(y_test, test_out, threshold=opt_thr)
        extra_ckpt["threshold"] = opt_thr
        out_key = "test_prob"
        LOGGER.info(
            "GNN [%s] Best Epoch=%d | Val ROC-AUC=%.4f PR-AUC=%.4f | Test ROC-AUC=%.4f PR-AUC=%.4f MCC=%.4f",
            ds.name.upper(),
            best_epoch,
            val_metrics["roc_auc"],
            val_metrics["pr_auc"],
            test_metrics["roc_auc"],
            test_metrics["pr_auc"],
            test_metrics["mcc"],
        )
    else:
        val_metrics = evaluate_regression_metrics(y_val, val_out)
        test_metrics = evaluate_regression_metrics(y_test, test_out)
        out_key = "test_pred"
        LOGGER.info(
            "GNN [%s] Best Epoch=%d | Val RMSE=%.4f R2=%.4f | Test RMSE=%.4f MAE=%.4f R2=%.4f Spearman=%.4f",
            ds.name.upper(),
            best_epoch,
            val_metrics["rmse"],
            val_metrics["r2"],
            test_metrics["rmse"],
            test_metrics["mae"],
            test_metrics["r2"],
            test_metrics["spearman_r"],
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(cfg),
            "desc_scaler_mean": ds.scaler.mean_.tolist(),
            "desc_scaler_scale": ds.scaler.scale_.tolist(),
            "descriptor_names": ds.descriptor_names,
            "best_epoch": best_epoch,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            **extra_ckpt,
        },
        checkpoint_path,
    )
    LOGGER.info("Saved GNN checkpoint -> %s", checkpoint_path)

    return {
        "val": val_metrics,
        "test": test_metrics,
        "y_test": y_test,
        out_key: test_out,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
    }


def train_gnn_bbbp(
    data_dir: Path = Path("data/processed"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate GNN_BBBP on the BBBP Bemis-Murcko scaffold split."""
    ds = load_scaffold_split_data(data_dir / "bbbp_features.npz", dataset_name="bbbp")
    return _train_gnn_core(
        ds=ds,
        task="classification",
        checkpoint_path=checkpoint_dir / "best_gnn_bbbp.pt",
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
    )


def train_gnn_ache(
    data_dir: Path = Path("data/processed"),
    checkpoint_dir: Path = Path("models/checkpoints"),
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate GNN_AChE on the CHEMBL220 Bemis-Murcko scaffold split."""
    ds = load_scaffold_split_data(data_dir / "chembl220_features.npz", dataset_name="chembl220")
    return _train_gnn_core(
        ds=ds,
        task="regression",
        checkpoint_path=checkpoint_dir / "best_gnn_ache.pt",
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
    )
