"""
LNMGAT model components.

This module contains the reusable model code for LNMGAT, including:
1. LapRLS fold-local score propagation.
2. k-nearest-neighbor graph construction from similarity matrices.
3. The dual-branch graph attention network used for drug-target interaction scoring.

All comments and runtime messages are written in English to keep the public
repository consistent and reproducible.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

try:
    from torch_geometric.nn import GATConv
except ImportError as exc:  # pragma: no cover - environment-dependent import
    raise ImportError(
        "torch_geometric is required for LNMGAT. Install a PyTorch Geometric build "
        "compatible with your local PyTorch and CUDA versions."
    ) from exc


Pair = Tuple[int, int]


@dataclass
class LNMGATConfig:
    """Default hyperparameters used by the LNMGAT reproduction scripts."""

    hidden_dim: int = 256
    heads: int = 8
    knn_k: int = 20
    dropout: float = 0.3
    lr: float = 5e-4
    epochs: int = 300
    batch_size: int = 512
    seed: int = 42
    n_splits: int = 5
    lambda_reg: float = 0.01
    neg_ratio: int = 2
    weight_gat_ws: float = 1.0
    weight_gat_cs: float = 1.0
    neg_eval_ratio: int = 3
    eval_sample_size: int = 2000
    weight_decay: float = 5e-4
    grad_clip_norm: float = 1.0
    f1_threshold: float = 0.5
    predict_batch_size: int = 256

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "LNMGATConfig":
        """Create a config object while ignoring unknown keys."""
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in values.items() if k in valid_keys}
        return cls(**filtered)


class LapRLS:
    """Laplacian-regularized least-squares scoring for DTI matrices.

    The implementation follows the fold-local setting used in the manuscript:
    held-out interactions are masked before LapRLS scores are computed, thereby
    avoiding label leakage from the test region.
    """

    def __init__(self, drug_sim_1: np.ndarray, drug_sim_2: np.ndarray, target_sim_1: np.ndarray, target_sim_2: np.ndarray):
        self.drug_sim_1 = drug_sim_1.astype(np.float32)
        self.drug_sim_2 = drug_sim_2.astype(np.float32)
        self.target_sim_1 = target_sim_1.astype(np.float32)
        self.target_sim_2 = target_sim_2.astype(np.float32)

    @staticmethod
    def _safe_inverse_sqrt_degree(similarity: np.ndarray) -> np.ndarray:
        degree = np.diag(similarity.sum(axis=1))
        eye = np.eye(degree.shape[0], dtype=np.float32)
        return fractional_matrix_power(degree + 1e-6 * eye, -0.5).astype(np.float32)

    def predict(self, interaction_matrix: np.ndarray) -> np.ndarray:
        """Return a normalized LapRLS score matrix with shape [n_drugs, n_targets]."""
        drug_sim = 0.5 * (self.drug_sim_1 + self.drug_sim_2)
        target_sim = 0.5 * (self.target_sim_1 + self.target_sim_2)

        drug_degree_inv_sqrt = self._safe_inverse_sqrt_degree(drug_sim)
        target_degree_inv_sqrt = self._safe_inverse_sqrt_degree(target_sim)

        drug_sim_norm = drug_degree_inv_sqrt @ drug_sim @ drug_degree_inv_sqrt
        target_sim_norm = target_degree_inv_sqrt @ target_sim @ target_degree_inv_sqrt

        drug_eye = np.eye(drug_sim_norm.shape[0], dtype=np.float32)
        target_eye = np.eye(target_sim_norm.shape[0], dtype=np.float32)
        drug_laplacian = drug_eye - drug_sim_norm
        target_laplacian = target_eye - target_sim_norm

        drug_inverse = np.linalg.inv(drug_sim_norm + 0.1 * drug_laplacian @ drug_sim_norm + 1e-6 * drug_eye)
        target_inverse = np.linalg.inv(target_sim_norm + 0.1 * target_laplacian @ target_sim_norm + 1e-6 * target_eye)

        drug_side_scores = drug_sim_norm @ drug_inverse @ interaction_matrix
        target_side_scores = target_sim_norm @ target_inverse @ interaction_matrix.T
        scores = (drug_side_scores + target_side_scores.T) / 2.0

        scaler = MinMaxScaler()
        return scaler.fit_transform(scores.ravel().reshape(-1, 1)).reshape(scores.shape).astype(np.float32)


class LNMGAT(nn.Module):
    """Dual-branch GAT model for drug-target interaction prediction."""

    def __init__(self, n_drugs: int, n_targets: int, hidden_dim: int = 256, heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.n_drugs = n_drugs
        self.n_targets = n_targets
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout

        self.drug_projection = nn.Linear(n_drugs, hidden_dim)
        self.target_projection = nn.Linear(n_targets, hidden_dim)
        self.drug_layer_norm = nn.LayerNorm(hidden_dim)
        self.target_layer_norm = nn.LayerNorm(hidden_dim)

        self.drug_gat_1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.drug_gat_2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.drug_gat_3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)

        self.target_gat_1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.target_gat_2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.target_gat_3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)

        self.drug_layer_norm_1 = nn.LayerNorm(hidden_dim * heads)
        self.drug_layer_norm_2 = nn.LayerNorm(hidden_dim * heads)
        self.target_layer_norm_1 = nn.LayerNorm(hidden_dim * heads)
        self.target_layer_norm_2 = nn.LayerNorm(hidden_dim * heads)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def encode(
        self,
        drug_features: torch.Tensor,
        drug_edges: torch.Tensor,
        target_features: torch.Tensor,
        target_edges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode drug and target nodes independently using two GAT branches."""
        drug_hidden = self.drug_layer_norm(self.drug_projection(drug_features))
        target_hidden = self.target_layer_norm(self.target_projection(target_features))

        drug_gat_1 = self.drug_layer_norm_1(torch.relu(self.drug_gat_1(drug_hidden, drug_edges)))
        drug_gat_2 = self.drug_layer_norm_2(torch.relu(self.drug_gat_2(drug_gat_1, drug_edges))) + drug_gat_1
        drug_embedding = self.drug_gat_3(drug_gat_2, drug_edges)

        target_gat_1 = self.target_layer_norm_1(torch.relu(self.target_gat_1(target_hidden, target_edges)))
        target_gat_2 = self.target_layer_norm_2(torch.relu(self.target_gat_2(target_gat_1, target_edges))) + target_gat_1
        target_embedding = self.target_gat_3(target_gat_2, target_edges)

        return drug_embedding, target_embedding

    def forward(
        self,
        drug_features: torch.Tensor,
        drug_edges: torch.Tensor,
        target_features: torch.Tensor,
        target_edges: torch.Tensor,
        drug_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for a batch of drug-target pairs."""
        drug_embedding, target_embedding = self.encode(drug_features, drug_edges, target_features, target_edges)
        pair_embedding = torch.cat([drug_embedding[drug_indices], target_embedding[target_indices]], dim=-1)
        return self.classifier(pair_embedding).squeeze(-1)

    def predict_full(
        self,
        drug_features: torch.Tensor,
        drug_edges: torch.Tensor,
        target_features: torch.Tensor,
        target_edges: torch.Tensor,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return full drug-target probability scores without materializing all pairs at once."""
        self.eval()
        with torch.no_grad():
            drug_embedding, target_embedding = self.encode(drug_features, drug_edges, target_features, target_edges)
            n_drugs = drug_embedding.size(0)
            n_targets = target_embedding.size(0)
            hidden_size = drug_embedding.size(1)
            device = drug_embedding.device
            output = torch.empty((n_drugs, n_targets), device=device, dtype=torch.float32)

            for start in range(0, n_drugs, batch_size):
                end = min(start + batch_size, n_drugs)
                drug_batch = drug_embedding[start:end]
                drug_expanded = drug_batch.unsqueeze(1).expand(-1, n_targets, -1)
                target_expanded = target_embedding.unsqueeze(0).expand(drug_expanded.size(0), -1, -1)
                pair_embedding = torch.cat([drug_expanded, target_expanded], dim=-1).reshape(-1, 2 * hidden_size)
                logits = self.classifier(pair_embedding)
                output[start:end] = torch.sigmoid(logits).view(drug_expanded.size(0), n_targets)

            return output.cpu().numpy().astype(np.float32)


# Backward-compatible alias for older checkpoints that may refer to LRGAT.
LRGAT = LNMGAT


def set_random_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_csv_matrix(path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load a CSV matrix whose first row and first column contain identifiers."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, header=0, index_col=0)
    return frame.to_numpy(dtype=np.float32), frame.index.astype(str).tolist(), frame.columns.astype(str).tolist()


def row_minmax_norm(matrix: np.ndarray) -> np.ndarray:
    """Apply row-wise min-max normalization without using global column statistics."""
    matrix = matrix.astype(np.float32)
    mins = matrix.min(axis=1, keepdims=True)
    maxs = matrix.max(axis=1, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return ((matrix - mins) / ranges).astype(np.float32)


def build_knn_graph(similarity: np.ndarray, k: int = 10, device: str | torch.device = "cpu") -> torch.Tensor:
    """Build a kNN edge index tensor from a precomputed similarity matrix."""
    similarity = np.clip(similarity.astype(np.float32), 0.0, 1.0)
    k = min(k, similarity.shape[0] - 1)
    if k < 1:
        k = 1
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    knn = kneighbors_graph(distance, k, mode="connectivity", include_self=False, metric="precomputed")
    coo = knn.tocoo()
    edge_index = np.vstack((coo.row, coo.col))
    return torch.tensor(edge_index, dtype=torch.long, device=device)


def evaluate_binary_scores(
    y_true: Sequence[float],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AUPR, AUROC, F1, precision, and recall for binary labels."""
    y_true_np = np.asarray(y_true).astype(int)
    y_score_np = np.asarray(y_score).astype(float)
    pos_count = int(np.sum(y_true_np == 1))
    neg_count = int(np.sum(y_true_np == 0))

    if pos_count == 0 or neg_count == 0:
        return {
            "AUPR": np.nan,
            "AUROC": np.nan,
            "F1": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "Positives": pos_count,
            "Negatives": neg_count,
        }

    y_pred = (y_score_np >= threshold).astype(int)
    return {
        "AUPR": float(average_precision_score(y_true_np, y_score_np)),
        "AUROC": float(roc_auc_score(y_true_np, y_score_np)),
        "F1": float(f1_score(y_true_np, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true_np, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true_np, y_pred, zero_division=0)),
        "Positives": pos_count,
        "Negatives": neg_count,
    }


def load_dti_data(
    data_dir: str,
    dataset: str,
    label: str,
    drug_sim_file_1: str | None = None,
    drug_sim_file_2: str | None = None,
    target_sim_file_1: str | None = None,
    target_sim_file_2: str | None = None,
    interaction_file: str | None = None,
) -> Dict[str, object]:
    """Load all matrices required by LNMGAT."""
    drug_sim_file_1 = drug_sim_file_1 or f"drug_similarity_{dataset}.csv"
    drug_sim_file_2 = drug_sim_file_2 or f"drug_IS_{label}_{dataset}.csv"
    target_sim_file_1 = target_sim_file_1 or f"target_similarity_{dataset}.csv"
    target_sim_file_2 = target_sim_file_2 or f"target_IS_{label}_{dataset}.csv"
    interaction_file = interaction_file or f"interaction_{label}_{dataset}.csv"

    drug_sim_1, drug_ids, _ = load_csv_matrix(os.path.join(data_dir, drug_sim_file_1))
    drug_sim_2, _, _ = load_csv_matrix(os.path.join(data_dir, drug_sim_file_2))
    target_sim_1, target_ids, _ = load_csv_matrix(os.path.join(data_dir, target_sim_file_1))
    target_sim_2, _, _ = load_csv_matrix(os.path.join(data_dir, target_sim_file_2))
    interaction_matrix, interaction_drug_ids, interaction_target_ids = load_csv_matrix(os.path.join(data_dir, interaction_file))

    if len(interaction_drug_ids) == len(drug_ids):
        drug_ids = interaction_drug_ids
    if len(interaction_target_ids) == len(target_ids):
        target_ids = interaction_target_ids

    return {
        "drug_sim_1": drug_sim_1,
        "drug_sim_2": drug_sim_2,
        "target_sim_1": target_sim_1,
        "target_sim_2": target_sim_2,
        "interaction_matrix": interaction_matrix.astype(np.float32),
        "drug_ids": drug_ids,
        "target_ids": target_ids,
    }


def prepare_full_graph_tensors(
    drug_sim_1: np.ndarray,
    drug_sim_2: np.ndarray,
    target_sim_1: np.ndarray,
    target_sim_2: np.ndarray,
    config: LNMGATConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare full-graph similarity features and kNN edges."""
    n_drugs = drug_sim_1.shape[0]
    n_targets = target_sim_1.shape[0]
    drug_similarity = 0.5 * (drug_sim_1 + drug_sim_2) + config.lambda_reg * np.eye(n_drugs, dtype=np.float32)
    target_similarity = 0.5 * (target_sim_1 + target_sim_2) + config.lambda_reg * np.eye(n_targets, dtype=np.float32)

    drug_features = torch.tensor(row_minmax_norm(drug_similarity), dtype=torch.float32, device=device)
    target_features = torch.tensor(row_minmax_norm(target_similarity), dtype=torch.float32, device=device)
    drug_edges = build_knn_graph(drug_similarity, config.knn_k, device)
    target_edges = build_knn_graph(target_similarity, config.knn_k, device)

    return drug_similarity, target_similarity, drug_features, target_features, drug_edges, target_edges


def sample_low_score_negatives(
    candidate_pairs: Sequence[Pair],
    score_matrix: np.ndarray,
    max_count: int,
) -> List[Pair]:
    """Select candidate pairs with the lowest LapRLS scores as reliable pseudo-negatives."""
    if max_count <= 0 or len(candidate_pairs) == 0:
        return []
    drug_indices = [pair[0] for pair in candidate_pairs]
    target_indices = [pair[1] for pair in candidate_pairs]
    scores = score_matrix[drug_indices, target_indices]
    selected = np.argsort(scores)[: min(max_count, len(candidate_pairs))]
    return [candidate_pairs[int(index)] for index in selected]


def save_prediction_tables(
    prediction_rows: List[Dict[str, object]],
    drug_ids: Sequence[str],
    target_ids: Sequence[str],
    output_dir: str,
    dataset: str,
    prefix: str = "LNMGAT",
) -> None:
    """Save scenario-specific prediction score files."""
    if not prediction_rows:
        print("No prediction rows were collected; prediction tables were not written.")
        return

    os.makedirs(output_dir, exist_ok=True)
    predictions = pd.DataFrame(prediction_rows)
    averaged = predictions.groupby(["Drug_idx", "Target_idx", "Scenario"], as_index=False)["Fused_Score"].mean()
    averaged["drug_id"] = averaged["Drug_idx"].apply(lambda index: drug_ids[int(index)])
    averaged["target_id"] = averaged["Target_idx"].apply(lambda index: target_ids[int(index)])
    averaged["drug-target"] = averaged["drug_id"] + "-" + averaged["target_id"]

    scenario_names = {
        "WS": "Warm-Start",
        "ND": "New-Drug",
        "NT": "New-Target",
        "NDNT": "New-Both",
    }
    output_columns = ["drug-target", "drug_id", "target_id", "Fused_Score"]

    for scenario_key, scenario_frame in averaged.groupby("Scenario"):
        scenario_name = scenario_names.get(str(scenario_key), str(scenario_key))
        output_path = os.path.join(output_dir, f"{prefix}_Predictions_{dataset}_{scenario_name}.csv")
        scenario_frame[output_columns].sort_values("Fused_Score", ascending=False).to_csv(output_path, index=False)
        print(f"Prediction scores saved to: {output_path}")
