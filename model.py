# LNMGAT — LapRLS-Driven Negative Mining Graph Attention Network

import os
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
# Import F1-related metrics
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import random
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================= Configuration (V15 optimized & patched) =============================
DATASET = 'davis'
LABEL = '0'
BASE_PATH = "./data_DTI"
OUT_PATH = './output'
DRUG_SIM_FILES = [f"drug_similarity_{DATASET}.csv", f"drug_IS_{LABEL}_{DATASET}.csv"]
TARGET_SIM_FILES = [f"target_similarity_{DATASET}.csv", f"target_IS_{LABEL}_{DATASET}.csv"]
INTERACTION_FILE = f"interaction_{LABEL}_{DATASET}.csv"
CONFIG = {
    'hidden_dim': 256,
    'heads': 8,
    'knn_k': 20,
    'dropout': 0.3,
    'lr': 0.0005,
    'epochs': 300,
    'batch_size': 512,
    'seed': 42,
    'n_splits': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'lambda_reg': 0.01,
    'neg_ratio': 2,
    'weight_gat_ws': 1,
    'weight_gat_cs': 1,
    'neg_eval_ratio': 3,
    'eval_sample_size': 2000,
    'weight_decay': 5e-4,
    'grad_clip_norm': 1.0,
    # Added F1 evaluation threshold
    'f1_threshold': 0.5,
}


# ============================= LapRLS (unchanged) =============================
class LapRLS:
    def __init__(self, Sd1, Sd2, St1, St2):
        self.Sd1, self.Sd2 = Sd1.astype(np.float32), Sd2.astype(np.float32)
        self.St1, self.St2 = St1.astype(np.float32), St2.astype(np.float32)

    def predict(self, A):
        Sd = 0.5 * (self.Sd1 + self.Sd2)
        St = 0.5 * (self.St1 + self.St2)
        # Degree matrices
        Dd = np.diag(Sd.sum(1))
        Dt = np.diag(St.sum(1))
        # Numerical safety checks
        if Dd.shape[0] != Sd.shape[0]:
            Dd = np.diag(np.sum(Sd, axis=1))
        if Dt.shape[0] != St.shape[0]:
            Dt = np.diag(np.sum(St, axis=1))

        Dd_inv_sqrt = fractional_matrix_power(Dd + 1e-6 * np.eye(Dd.shape[0], dtype=np.float32), -0.5)
        Dt_inv_sqrt = fractional_matrix_power(Dt + 1e-6 * np.eye(Dt.shape[0], dtype=np.float32), -0.5)
        Sd_n = Dd_inv_sqrt @ Sd @ Dd_inv_sqrt
        St_n = Dt_inv_sqrt @ St @ Dt_inv_sqrt
        Id = np.eye(Sd_n.shape[0], dtype=np.float32)
        It = np.eye(St_n.shape[0], dtype=np.float32)
        Ld = Id - Sd_n
        Lt = It - St_n

        # Stable inversion with regularization
        inv_d = np.linalg.inv(Sd_n + 0.1 * Ld @ Sd_n + 1e-6 * Id)
        inv_t = np.linalg.inv(St_n + 0.1 * Lt @ St_n + 1e-6 * It)

        # A is expected to be of shape (n_drugs, n_targets)
        Fd = Sd_n @ inv_d @ A
        Ft = St_n @ inv_t @ A.T
        F = (Fd + Ft.T) / 2.0

        # Global min-max scaling (alternative: row-wise)
        scaler = MinMaxScaler()
        F_scaled = scaler.fit_transform(F.ravel().reshape(-1, 1)).reshape(F.shape).astype(np.float32)
        return F_scaled


# ============================= Model =============================
class LRGAT(nn.Module):
    def __init__(self, n_drugs, n_targets, hidden_dim=CONFIG['hidden_dim'], heads=CONFIG['heads'],
                 dropout=CONFIG['dropout']):
        super().__init__()
        # NOTE: inputs are similarity rows of length n_drugs / n_targets respectively
        self.d_proj = nn.Linear(n_drugs, hidden_dim)
        self.t_proj = nn.Linear(n_targets, hidden_dim)
        self.ln_d = nn.LayerNorm(hidden_dim)
        self.ln_t = nn.LayerNorm(hidden_dim)
        self.gat1_d = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2_d = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat3_d = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.gat1_t = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2_t = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat3_t = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)

        self.ln_d_1 = nn.LayerNorm(hidden_dim * heads)
        self.ln_d_2 = nn.LayerNorm(hidden_dim * heads)
        self.ln_t_1 = nn.LayerNorm(hidden_dim * heads)
        self.ln_t_2 = nn.LayerNorm(hidden_dim * heads)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

    def encode(self, dx, de, tx, te):
        xd = self.ln_d(self.d_proj(dx))
        xt = self.ln_t(self.t_proj(tx))

        x1_d = self.ln_d_1(torch.relu(self.gat1_d(xd, de)))
        x2_d = self.ln_d_2(torch.relu(self.gat2_d(x1_d, de))) + x1_d
        xd = self.gat3_d(x2_d, de)

        x1_t = self.ln_t_1(torch.relu(self.gat1_t(xt, te)))
        x2_t = self.ln_t_2(torch.relu(self.gat2_t(x1_t, te))) + x1_t
        xt = self.gat3_t(x2_t, te)

        return xd, xt

    def forward(self, dx, de, tx, te, di, ti):
        xd, xt = self.encode(dx, de, tx, te)
        return self.classifier(torch.cat([xd[di], xt[ti]], dim=-1)).squeeze(-1)

    def predict_full(self, dx, de, tx, te, batch_size=256):
        """
        Batchified full prediction to avoid creating an N_d * N_t matrix at once.
        Returns numpy array shape (N_d, N_t)
        """
        with torch.no_grad():
            xd, xt = self.encode(dx, de, tx, te)
            N_d, H = xd.size(0), xd.size(1)
            N_t = xt.size(0)
            device = xd.device
            out = torch.empty((N_d, N_t), device=device, dtype=torch.float32)
            # Iterate over drugs in batches
            for i in range(0, N_d, batch_size):
                i2 = min(i + batch_size, N_d)
                xd_batch = xd[i:i2]  # (b, H)
                # Expand: (b, N_t, H)
                xd_exp = xd_batch.unsqueeze(1).expand(-1, N_t, -1)
                xt_exp = xt.unsqueeze(0).expand(xd_exp.size(0), -1, -1)
                pair = torch.cat([xd_exp, xt_exp], dim=-1).reshape(-1, 2 * H)
                logits = self.classifier(pair)
                probs = torch.sigmoid(logits).view(xd_exp.size(0), N_t)
                out[i:i2] = probs
            return out.cpu().numpy().astype(np.float32)


# ============================= Utilities =============================
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, header=0, index_col=0)
    return df.to_numpy(dtype=np.float32), df.index.tolist(), df.columns.tolist()


def row_minmax_norm(M):
    """Row-wise min-max normalization. Avoids using global column statistics."""
    M = M.astype(np.float32)
    mins = M.min(axis=1, keepdims=True)
    maxs = M.max(axis=1, keepdims=True)
    rng = (maxs - mins)
    rng[rng == 0] = 1.0
    return ((M - mins) / rng).astype(np.float32)


def build_knn(S, k=10, dev='cpu'):
    """
    Build kNN graph with precomputed 'distance' = 1 - similarity.
    NOTE: include_self=False to avoid double self-loops for GATConv.
    Returns edge_index tensor shape (2, E) on device 'dev'.
    """
    S = np.clip(S, 0, 1)
    k = min(k, S.shape[0] - 1)
    if k < 1: k = 1
    dist = 1.0 - S
    np.fill_diagonal(dist, 0)
    knn = kneighbors_graph(dist, k, mode='connectivity', include_self=False, metric='precomputed')
    coo = knn.tocoo()
    return torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=dev)


def evaluate(y_true, y_pred, name):
    """
    Evaluation function: computes AUPR, AUROC, Precision, Recall, and F1-score.
    """
    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)

    if pos_count == 0 or neg_count == 0:
        print(f" {name:38}: AUPR = N/A | AUROC = N/A | F1 = N/A (Pos={pos_count}, Neg={neg_count})")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    aupr = average_precision_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)

    threshold = CONFIG['f1_threshold']
    y_pred_binary = (y_pred >= threshold).astype(int)

    # F1, Precision, Recall
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)

    print(
        f" {name:38}: AUPR = {aupr:.4f} | AUROC = {auroc:.4f} | F1 = {f1:.4f} (P={precision:.4f}, R={recall:.4f}) (Pos={pos_count}, Neg={neg_count})")

    return aupr, auroc, f1, precision, recall


# ============================= Main =============================
def main():
    seed = CONFIG['seed']
    device = CONFIG['device']

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("=" * 80)
    print(" LRGAT: Laplacian-Regularized GAT (V15 - Patch Fixed)")
    print(f" **Evaluation Strategy: LapRLS Lowest Score Negatives ({CONFIG['neg_eval_ratio']}:1 Ratio), Fold-local LapRLS**")
    print(f" **F1/P/R Threshold: {CONFIG['f1_threshold']}**")
    print("=" * 80)

    # 1. Data loading and preprocessing
    try:
        Sd1, _, _ = load_csv(os.path.join(BASE_PATH, DRUG_SIM_FILES[0]))
        Sd2, _, _ = load_csv(os.path.join(BASE_PATH, DRUG_SIM_FILES[1]))
        St1, _, _ = load_csv(os.path.join(BASE_PATH, TARGET_SIM_FILES[0]))
        St2, _, _ = load_csv(os.path.join(BASE_PATH, TARGET_SIM_FILES[1]))
        A_full, drug_ids, target_ids = load_csv(os.path.join(BASE_PATH, INTERACTION_FILE))
        print("Real data loaded successfully")
    except Exception as e:
        print(f"Data loading failed ({e}). Running with simulated data (Test Mode)")
        n_d, n_t = 68, 442
        A_full = np.random.randint(0, 2, (n_d, n_t)).astype(np.float32)
        Sd1 = Sd2 = np.eye(n_d, dtype=np.float32)
        St1 = St2 = np.eye(n_t, dtype=np.float32)
        drug_ids = [f"D{i}" for i in range(n_d)]
        target_ids = [f"T{i}" for i in range(n_t)]

    n_drugs, n_targets = A_full.shape

    # GAT features (row-normalized to avoid global column leakage)
    Sd_full = 0.5 * (Sd1 + Sd2) + CONFIG['lambda_reg'] * np.eye(n_drugs, dtype=np.float32)
    St_full = 0.5 * (St1 + St2) + CONFIG['lambda_reg'] * np.eye(n_targets, dtype=np.float32)
    drug_x_full = torch.tensor(row_minmax_norm(Sd_full), dtype=torch.float).to(device)
    target_x_full = torch.tensor(row_minmax_norm(St_full), dtype=torch.float).to(device)
    edge_d_full = build_knn(Sd_full, CONFIG['knn_k'], device)
    edge_t_full = build_knn(St_full, CONFIG['knn_k'], device)

    # 2. KFold split preparation
    all_pos_pairs = np.argwhere(A_full == 1).tolist()
    pos_indices = np.arange(len(all_pos_pairs))
    n_pos_total = len(all_pos_pairs)
    all_pos_pairs_array = np.array(all_pos_pairs)
    print(f"Total number of positive samples: {n_pos_total}")

    # KFold
    kf_d = KFold(CONFIG['n_splits'], shuffle=True, random_state=seed)
    kf_t = KFold(CONFIG['n_splits'], shuffle=True, random_state=seed + 1)
    kf_pos = KFold(CONFIG['n_splits'], shuffle=True, random_state=seed + 2)

    global_model = LRGAT(n_drugs, n_targets).to(device)
    global_model_state = global_model.state_dict()
    all_metrics = []  # Store metrics for each fold
    gat_preds_list = []
    lap_preds_list = []

    # Store predictions per fold and scenario
    fold_predictions = []

    iterators = zip(kf_d.split(range(n_drugs)), kf_t.split(range(n_targets)), kf_pos.split(pos_indices))

    for fold, ((tr_d_i, te_d_i), (tr_t_i, te_t_i), (tr_pos_i, te_pos_i)) in enumerate(iterators):
        print(f"\nFold {fold + 1}/{CONFIG['n_splits']}")

        train_d_idx, te_d_idx = tr_d_i, te_d_i
        train_t_idx, te_t_idx = tr_t_i, te_t_i

        # 4.1. Split WS training and test sets (based on pos split)
        pos_pairs_train = [
            (d, t) for d, t in all_pos_pairs_array[tr_pos_i]
            if (d in train_d_idx) and (t in train_t_idx)
        ]
        ws_test_pos_pairs = [
            (d, t) for d, t in all_pos_pairs_array[te_pos_i]
            if (d in train_d_idx) and (t in train_t_idx)
        ]

        # 4.2. Build A_train (mask test regions and WS test points)
        A_train = A_full.copy()

        # Mask cold-start regions
        for d in te_d_idx:
            A_train[d, :] = 0.0
        for t in te_t_idx:
            A_train[:, t] = 0.0
        # Mask WS test positive pairs
        for (d, t) in ws_test_pos_pairs:
            A_train[d, t] = 0.0

        # Fold-local LapRLS prediction
        print("Computing LapRLS prediction for current fold...")
        laprls_fold = LapRLS(Sd1, Sd2, St1, St2)
        F_lap_fold = laprls_fold.predict(A_train)
        lap_preds_list.append(F_lap_fold)

        # 4.3. Build training negative sample pool (only in train_d x train_t region)
        all_tr_unknown_pairs = [
            (d, t) for d in train_d_idx for t in train_t_idx
            if A_full[d, t] == 0 and (d, t) not in ws_test_pos_pairs
        ]

        train_neg = []
        if len(all_tr_unknown_pairs) > 0:
            unknown_scores_tr = F_lap_fold[[i for i, j in all_tr_unknown_pairs], [j for i, j in all_tr_unknown_pairs]]
            n_pos = len(pos_pairs_train)
            n_neg_ratio_limit = n_pos * CONFIG['neg_ratio']
            n_neg_max_limit = len(all_tr_unknown_pairs)
            n_neg_needed = int(min(n_neg_ratio_limit, n_neg_max_limit))
            sample_k = min(n_neg_needed, len(unknown_scores_tr))

            if sample_k > 0:
                # Select lowest-score (most confident negatives)
                low_idx = np.argsort(unknown_scores_tr)[:sample_k]
                train_neg = [all_tr_unknown_pairs[i] for i in low_idx]
            else:
                # Fallback: random sample
                fallback_k = min(max(1, int(n_pos * 0.5)), len(all_tr_unknown_pairs))
                train_neg = random.sample(all_tr_unknown_pairs, k=fallback_k)
        else:
            print("Warning: No unknown interactions in training region (all_tr_unknown_pairs == 0)")

        print(f"Training set ready: Pos={len(pos_pairs_train)} | Neg={len(train_neg)}")

        # 4.4. Model and data loader
        model = LRGAT(n_drugs, n_targets).to(device)
        model.load_state_dict(global_model_state)

        # GAT subgraph & features (train-only)
        Sd_tr_sub = Sd_full[np.ix_(train_d_idx, train_d_idx)]
        St_tr_sub = St_full[np.ix_(train_t_idx, train_t_idx)]
        edge_d_tr = build_knn(Sd_tr_sub, CONFIG['knn_k'], device)
        edge_t_tr = build_knn(St_tr_sub, CONFIG['knn_k'], device)
        train_drug_x = drug_x_full[train_d_idx]
        train_target_x = target_x_full[train_t_idx]

        global_to_local_d = {g: l for l, g in enumerate(train_d_idx)}
        global_to_local_t = {g: l for l, g in enumerate(train_t_idx)}

        samples = [(d, t, 1.0) for d, t in pos_pairs_train] + [(d, t, 0.0) for d, t in train_neg]

        if len(samples) == 0:
            print("Warning: No training samples in current fold, skipping this fold")
            continue

        loader = torch.utils.data.DataLoader(samples, batch_size=CONFIG['batch_size'], shuffle=True,
                                             collate_fn=lambda x: (
                                                 torch.tensor([global_to_local_d[i[0]] for i in x], dtype=torch.long),
                                                 torch.tensor([global_to_local_t[i[1]] for i in x], dtype=torch.long),
                                                 torch.tensor([i[2] for i in x], dtype=torch.float)
                                             ))
        opt = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG['epochs'])
        criterion = nn.BCEWithLogitsLoss()

        # 4.5. Training
        model.train()
        epoch_bar = tqdm(range(CONFIG['epochs']), desc=f"Fold {fold + 1}", leave=False)
        for epoch in epoch_bar:
            total_loss = 0.0
            for db_local, tb_local, yb in loader:
                db_local, tb_local, yb = db_local.to(device), tb_local.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(train_drug_x, edge_d_tr, train_target_x, edge_t_tr, db_local, tb_local)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_norm'])
                opt.step()
                total_loss += loss.item()
            sch.step()
            avg_loss = total_loss / (len(loader) if len(loader) > 0 else 1)
            epoch_bar.set_postfix({'loss': f"{avg_loss:.4f}"})
        epoch_bar.close()

        # 5. Inference (full graph) and evaluation
        model.eval()
        F_gat = model.predict_full(drug_x_full, edge_d_full, target_x_full, edge_t_full, batch_size=256)
        gat_preds_list.append(F_gat)

        scenarios = {
            "Warm-Start (WS)": (train_d_idx, train_t_idx, ws_test_pos_pairs, "WS"),
            "Semi-Cold: New-Drug (ND)": (te_d_idx, train_t_idx, None, "ND"),
            "Semi-Cold: New-Target (NT)": (train_d_idx, te_t_idx, None, "NT"),
            "Full-Cold: New-Both (ND+NT)": (te_d_idx, te_t_idx, None, "NDNT")
        }

        metrics = {"Fold": fold + 1}

        current_fold_preds = []

        print(" Evaluation (Fold-local LapRLS & GAT fusion):")

        train_pos_set = set(pos_pairs_train)

        for name, (dset, tset, ws_pos_pairs_arg, scenario_key) in scenarios.items():
            w_gat = CONFIG['weight_gat_ws'] if name == "Warm-Start (WS)" else CONFIG['weight_gat_cs']

            # Determine all unknown pairs in the evaluation region
            all_pairs_scene = []
            if name == "Warm-Start (WS)":
                all_pairs_scene = [(d, t) for d in dset for t in tset]
            else:
                all_pairs_scene = [(d, t) for d in dset for t in tset]

            # Positive samples for evaluation
            if name == "Warm-Start (WS)":
                pos_pairs_scene = ws_pos_pairs_arg
            else:
                pos_pairs_scene = [(d, t) for d, t in all_pairs_scene if
                                   A_full[d, t] == 1 and (d, t) not in train_pos_set]

            unknown_pairs = [(d, t) for d, t in all_pairs_scene if A_full[d, t] == 0]

            if not pos_pairs_scene:
                print(f" {name:38}: N/A (No positive samples)")

                if name != "Warm-Start (WS)":
                    for d_idx, t_idx in all_pairs_scene:
                        fused_score = w_gat * F_gat[d_idx, t_idx] + (1 - w_gat) * F_lap_fold[d_idx, t_idx]
                        current_fold_preds.append({
                            'Fold': fold + 1,
                            'Drug_idx': d_idx,
                            'Target_idx': t_idx,
                            'Scenario': scenario_key,
                            'Fused_Score': fused_score
                        })
                continue

            # Sample positives for fair evaluation
            n_pos_limit = min(CONFIG['eval_sample_size'], len(pos_pairs_scene))
            pos_sample = random.sample(pos_pairs_scene, k=n_pos_limit) if len(pos_pairs_scene) >= n_pos_limit else list(pos_pairs_scene)

            # Negative sampling using LapRLS scores
            n_neg_needed = int(n_pos_limit * CONFIG['neg_eval_ratio'])
            if len(unknown_pairs) == 0:
                print(f" {name:38}: N/A (No unknown/negative pairs)")
                continue

            d_indices = [d for d, t in unknown_pairs]
            t_indices = [t for d, t in unknown_pairs]
            unknown_scores = F_lap_fold[d_indices, t_indices]
            sorted_indices = np.argsort(unknown_scores)
            n_select = min(n_neg_needed, len(unknown_scores))
            selected_indices = sorted_indices[:n_select] if n_select > 0 else []
            neg_sample = [unknown_pairs[i] for i in selected_indices] if len(selected_indices) > 0 else []

            if len(neg_sample) == 0:
                fallback_k = min(len(unknown_pairs), max(1, int(n_pos_limit * 0.5)))
                neg_sample = random.sample(unknown_pairs, k=fallback_k)

            all_pairs = neg_sample + pos_sample
            random.shuffle(all_pairs)
            pairs_d = [p[0] for p in all_pairs]
            pairs_t = [p[1] for p in all_pairs]
            y_true = A_full[pairs_d, pairs_t]

            y_pred = w_gat * F_gat[pairs_d, pairs_t] + (1 - w_gat) * F_lap_fold[pairs_d, pairs_t]

            aupr, auroc, f1, precision, recall = evaluate(y_true, y_pred, name)
            metrics[f"{name}_AUPR"] = aupr
            metrics[f"{name}_AUROC"] = auroc
            metrics[f"{name}_F1"] = f1
            metrics[f"{name}_Precision"] = precision
            metrics[f"{name}_Recall"] = recall

            # Record predictions for all unknown and test positive pairs
            prediction_pairs = pos_pairs_scene + unknown_pairs

            for d_idx, t_idx in prediction_pairs:
                fused_score = w_gat * F_gat[d_idx, t_idx] + (1 - w_gat) * F_lap_fold[d_idx, t_idx]

                if name == "Warm-Start (WS)" and (d_idx, t_idx) in train_pos_set:
                    continue

                current_fold_preds.append({
                    'Fold': fold + 1,
                    'Drug_idx': d_idx,
                    'Target_idx': t_idx,
                    'Scenario': scenario_key,
                    'Fused_Score': fused_score
                })

        all_metrics.append(metrics)
        fold_predictions.extend(current_fold_preds)

    # 6. Summary and saving results
    os.makedirs(OUT_PATH, exist_ok=True)

    # Save fold-wise metrics
    df_all_folds = pd.DataFrame(all_metrics)
    metrics_file_path = os.path.join(OUT_PATH, f"LRGAT_Fold_Metrics_{DATASET}.csv")
    df_all_folds.to_csv(metrics_file_path, index=False)
    print(f"\nFold-wise performance metrics saved to: {metrics_file_path}")

    # Compute and save mean metrics
    mean_metrics = {}
    std_metrics = {}
    metrics_to_summarize = ["AUPR", "AUROC", "F1", "Precision", "Recall"]
    scenarios = ["Warm-Start (WS)", "Semi-Cold: New-Drug (ND)", "Semi-Cold: New-Target (NT)",
                 "Full-Cold: New-Both (ND+NT)"]

    for metric in metrics_to_summarize:
        for scenario in scenarios:
            col = f"{scenario}_{metric}"
            if col in df_all_folds.columns:
                valid = df_all_folds[col].dropna()
                if not valid.empty:
                    mean_metrics[f"{scenario}_{metric}_Mean"] = valid.mean()
                    std_metrics[f"{scenario}_{metric}_Std"] = valid.std()

    df_mean = pd.DataFrame({
        'Scenario_Metric': [f"{s}_{m}" for s in scenarios for m in metrics_to_summarize],
        'Mean': [mean_metrics.get(f"{s}_{m}_Mean", np.nan) for s in scenarios for m in metrics_to_summarize],
        'Std': [std_metrics.get(f"{s}_{m}_Std", np.nan) for s in scenarios for m in metrics_to_summarize]
    })

    mean_metrics_file_path = os.path.join(OUT_PATH, f"LRGAT_Mean_Metrics_{DATASET}.csv")
    df_mean.to_csv(mean_metrics_file_path, index=False)
    print(f"Final average performance metrics saved to: {mean_metrics_file_path}")

    # Print final results
    print("\n" + "=" * 80)
    print("LRGAT FINAL RESULTS (Patch Fixed)")
    print("=" * 80)
    for metric in metrics_to_summarize:
        print(f"\n{metric}:")
        for scenario in scenarios:
            mean_key = f"{scenario}_{metric}_Mean"
            std_key = f"{scenario}_{metric}_Std"
            if mean_key in mean_metrics:
                print(f" {scenario:38}: {mean_metrics[mean_key]:.4f} ± {std_metrics[std_key]:.4f}")
            else:
                print(f" {scenario:38}: N/A")

    # Save scenario-specific prediction files
    if fold_predictions:
        df_preds = pd.DataFrame(fold_predictions)

        # Average predictions across folds for same (Drug, Target)
        df_preds_avg = df_preds.groupby(['Drug_idx', 'Target_idx', 'Scenario'])['Fused_Score'].mean().reset_index()

        # Convert to real IDs
        df_preds_avg['drug_id'] = df_preds_avg['Drug_idx'].apply(lambda i: drug_ids[i])
        df_preds_avg['target_id'] = df_preds_avg['Target_idx'].apply(lambda i: target_ids[i])
        df_preds_avg['drug-target'] = df_preds_avg['drug_id'] + '-' + df_preds_avg['target_id']

        scenario_names = {
            "WS": "Warm-Start", "ND": "New-Drug", "NT": "New-Target", "NDNT": "New-Both"
        }

        for scenario_key, scenario_df in df_preds_avg.groupby('Scenario'):
            file_name = f"LRGAT_Predictions_{DATASET}_{scenario_names[scenario_key]}.csv"
            output_cols = ['drug-target', 'drug_id', 'target_id', 'Fused_Score']
            scenario_df[output_cols].sort_values("Fused_Score", ascending=False).to_csv(
                os.path.join(OUT_PATH, file_name), index=False
            )
            print(f"\nLRGAT prediction scores ({scenario_names[scenario_key]}) saved to: {file_name}")
    else:
        print("\nWarning: Not enough prediction pairs collected, skipping scenario-specific prediction files.")


if __name__ == "__main__":
    main()