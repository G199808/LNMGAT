"""
Train and evaluate LNMGAT from scratch without writing model checkpoint files.

Example:
    python train.py --dataset davis --label 0 --data-dir ./data_DTI \
        --output-dir ./output --epochs 300
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm

from lnmgat_model import (
    LNMGAT,
    LNMGATConfig,
    LapRLS,
    build_knn_graph,
    evaluate_binary_scores,
    load_dti_data,
    prepare_full_graph_tensors,
    sample_low_score_negatives,
    save_prediction_tables,
    set_random_seed,
)

Pair = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LNMGAT with fold-local LapRLS negative mining.")
    parser.add_argument("--dataset", type=str, default="davis", help="Dataset name, e.g., davis, kiba, bindingdb.")
    parser.add_argument("--label", type=str, default="0", help="Interaction label suffix used in input filenames.")
    parser.add_argument("--data-dir", type=str, default="./data_DTI", help="Directory containing DTI matrices.")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory for metrics and prediction tables.")

    parser.add_argument("--drug-sim-file-1", type=str, default=None, help="Optional first drug similarity CSV filename.")
    parser.add_argument("--drug-sim-file-2", type=str, default=None, help="Optional second drug similarity CSV filename.")
    parser.add_argument("--target-sim-file-1", type=str, default=None, help="Optional first target similarity CSV filename.")
    parser.add_argument("--target-sim-file-2", type=str, default=None, help="Optional second target similarity CSV filename.")
    parser.add_argument("--interaction-file", type=str, default=None, help="Optional interaction matrix CSV filename.")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--neg-ratio", type=int, default=2)
    parser.add_argument("--weight-gat-ws", type=float, default=1.0)
    parser.add_argument("--weight-gat-cs", type=float, default=1.0)
    parser.add_argument("--neg-eval-ratio", type=int, default=3)
    parser.add_argument("--eval-sample-size", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--f1-threshold", type=float, default=0.5)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> LNMGATConfig:
    return LNMGATConfig(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        knn_k=args.knn_k,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        n_splits=args.n_splits,
        lambda_reg=args.lambda_reg,
        neg_ratio=args.neg_ratio,
        weight_gat_ws=args.weight_gat_ws,
        weight_gat_cs=args.weight_gat_cs,
        neg_eval_ratio=args.neg_eval_ratio,
        eval_sample_size=args.eval_sample_size,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        f1_threshold=args.f1_threshold,
        predict_batch_size=args.predict_batch_size,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_arg)


def evaluate_scenarios(
    fold: int,
    interaction_matrix: np.ndarray,
    train_drug_indices: np.ndarray,
    test_drug_indices: np.ndarray,
    train_target_indices: np.ndarray,
    test_target_indices: np.ndarray,
    train_positive_pairs: List[Pair],
    warm_start_test_positive_pairs: List[Pair],
    gat_scores: np.ndarray,
    lap_scores: np.ndarray,
    config: LNMGATConfig,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Evaluate all four scenarios and collect prediction rows."""
    scenarios = {
        "Warm-Start (WS)": (train_drug_indices, train_target_indices, warm_start_test_positive_pairs, "WS"),
        "Semi-Cold: New-Drug (ND)": (test_drug_indices, train_target_indices, None, "ND"),
        "Semi-Cold: New-Target (NT)": (train_drug_indices, test_target_indices, None, "NT"),
        "Full-Cold: New-Both (ND+NT)": (test_drug_indices, test_target_indices, None, "NDNT"),
    }

    metrics: Dict[str, float] = {"Fold": fold}
    prediction_rows: List[Dict[str, object]] = []
    train_positive_set = set(train_positive_pairs)

    for scenario_name, (drug_set, target_set, warm_start_pairs, scenario_key) in scenarios.items():
        gat_weight = config.weight_gat_ws if scenario_key == "WS" else config.weight_gat_cs
        all_pairs = [(int(drug_index), int(target_index)) for drug_index in drug_set for target_index in target_set]

        if scenario_key == "WS":
            positive_pairs = list(warm_start_pairs or [])
        else:
            positive_pairs = [pair for pair in all_pairs if interaction_matrix[pair[0], pair[1]] == 1 and pair not in train_positive_set]

        unknown_pairs = [pair for pair in all_pairs if interaction_matrix[pair[0], pair[1]] == 0]

        if not positive_pairs or not unknown_pairs:
            print(f"{scenario_name:38}: N/A (insufficient positive or unknown pairs)")
            continue

        n_pos = min(config.eval_sample_size, len(positive_pairs))
        positive_sample = random.sample(positive_pairs, k=n_pos) if len(positive_pairs) > n_pos else list(positive_pairs)
        negative_sample = sample_low_score_negatives(unknown_pairs, lap_scores, int(n_pos * config.neg_eval_ratio))
        if len(negative_sample) == 0:
            fallback_count = min(len(unknown_pairs), max(1, int(n_pos * 0.5)))
            negative_sample = random.sample(unknown_pairs, k=fallback_count)

        eval_pairs = negative_sample + positive_sample
        random.shuffle(eval_pairs)
        drug_eval = [pair[0] for pair in eval_pairs]
        target_eval = [pair[1] for pair in eval_pairs]
        y_true = interaction_matrix[drug_eval, target_eval]
        y_score = gat_weight * gat_scores[drug_eval, target_eval] + (1.0 - gat_weight) * lap_scores[drug_eval, target_eval]

        scenario_metrics = evaluate_binary_scores(y_true, y_score, config.f1_threshold)
        print(
            f"{scenario_name:38}: AUPR={scenario_metrics['AUPR']:.4f} | "
            f"AUROC={scenario_metrics['AUROC']:.4f} | F1={scenario_metrics['F1']:.4f} | "
            f"P={scenario_metrics['Precision']:.4f} | R={scenario_metrics['Recall']:.4f} | "
            f"Pos={scenario_metrics['Positives']} | Neg={scenario_metrics['Negatives']}"
        )

        for metric_name, metric_value in scenario_metrics.items():
            metrics[f"{scenario_name}_{metric_name}"] = metric_value

        for drug_idx, target_idx in positive_pairs + unknown_pairs:
            if scenario_key == "WS" and (drug_idx, target_idx) in train_positive_set:
                continue
            fused_score = float(gat_weight * gat_scores[drug_idx, target_idx] + (1.0 - gat_weight) * lap_scores[drug_idx, target_idx])
            prediction_rows.append(
                {
                    "Fold": fold,
                    "Drug_idx": int(drug_idx),
                    "Target_idx": int(target_idx),
                    "Scenario": scenario_key,
                    "Fused_Score": fused_score,
                }
            )

    return metrics, prediction_rows


def write_metric_summaries(all_metrics: List[Dict[str, float]], output_dir: str, dataset: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fold_metrics = pd.DataFrame(all_metrics)
    fold_path = os.path.join(output_dir, f"LNMGAT_Fold_Metrics_{dataset}.csv")
    fold_metrics.to_csv(fold_path, index=False)
    print(f"Fold-wise metrics saved to: {fold_path}")

    scenarios = ["Warm-Start (WS)", "Semi-Cold: New-Drug (ND)", "Semi-Cold: New-Target (NT)", "Full-Cold: New-Both (ND+NT)"]
    metrics = ["AUPR", "AUROC", "F1", "Precision", "Recall"]
    rows = []
    for scenario in scenarios:
        for metric in metrics:
            column = f"{scenario}_{metric}"
            if column in fold_metrics.columns:
                values = fold_metrics[column].dropna()
                rows.append(
                    {
                        "Scenario": scenario,
                        "Metric": metric,
                        "Mean": float(values.mean()) if not values.empty else np.nan,
                        "Std": float(values.std()) if len(values) > 1 else 0.0,
                    }
                )
    mean_path = os.path.join(output_dir, f"LNMGAT_Mean_Metrics_{dataset}.csv")
    pd.DataFrame(rows).to_csv(mean_path, index=False)
    print(f"Mean metrics saved to: {mean_path}")


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    device = resolve_device(args.device)
    set_random_seed(config.seed)

    print("=" * 80)
    print("LNMGAT training with fold-local LapRLS negative mining")
    print(f"Dataset: {args.dataset} | Label: {args.label} | Device: {device}")
    print("=" * 80)

    data = load_dti_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        label=args.label,
        drug_sim_file_1=args.drug_sim_file_1,
        drug_sim_file_2=args.drug_sim_file_2,
        target_sim_file_1=args.target_sim_file_1,
        target_sim_file_2=args.target_sim_file_2,
        interaction_file=args.interaction_file,
    )

    drug_sim_1 = data["drug_sim_1"]
    drug_sim_2 = data["drug_sim_2"]
    target_sim_1 = data["target_sim_1"]
    target_sim_2 = data["target_sim_2"]
    interaction_matrix = data["interaction_matrix"]
    drug_ids = data["drug_ids"]
    target_ids = data["target_ids"]

    n_drugs, n_targets = interaction_matrix.shape
    print(f"Loaded matrices: drugs={n_drugs}, targets={n_targets}, positives={int(interaction_matrix.sum())}")

    drug_similarity, target_similarity, drug_features, target_features, full_drug_edges, full_target_edges = prepare_full_graph_tensors(
        drug_sim_1, drug_sim_2, target_sim_1, target_sim_2, config, device
    )

    all_positive_pairs = np.argwhere(interaction_matrix == 1)
    positive_indices = np.arange(len(all_positive_pairs))

    kfold_drugs = KFold(config.n_splits, shuffle=True, random_state=config.seed)
    kfold_targets = KFold(config.n_splits, shuffle=True, random_state=config.seed + 1)
    kfold_positives = KFold(config.n_splits, shuffle=True, random_state=config.seed + 2)

    base_model = LNMGAT(n_drugs, n_targets, config.hidden_dim, config.heads, config.dropout).to(device)
    base_state_dict = base_model.state_dict()

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics: List[Dict[str, float]] = []
    all_prediction_rows: List[Dict[str, object]] = []

    fold_iterators = zip(
        kfold_drugs.split(range(n_drugs)),
        kfold_targets.split(range(n_targets)),
        kfold_positives.split(positive_indices),
    )

    for fold, ((train_drug_indices, test_drug_indices), (train_target_indices, test_target_indices), (train_pos_indices, test_pos_indices)) in enumerate(fold_iterators, start=1):
        print("\n" + "-" * 80)
        print(f"Fold {fold}/{config.n_splits}")

        train_positive_pairs = [
            (int(drug_idx), int(target_idx))
            for drug_idx, target_idx in all_positive_pairs[train_pos_indices]
            if drug_idx in train_drug_indices and target_idx in train_target_indices
        ]
        warm_start_test_positive_pairs = [
            (int(drug_idx), int(target_idx))
            for drug_idx, target_idx in all_positive_pairs[test_pos_indices]
            if drug_idx in train_drug_indices and target_idx in train_target_indices
        ]

        train_interaction_matrix = interaction_matrix.copy()
        for drug_idx in test_drug_indices:
            train_interaction_matrix[int(drug_idx), :] = 0.0
        for target_idx in test_target_indices:
            train_interaction_matrix[:, int(target_idx)] = 0.0
        for drug_idx, target_idx in warm_start_test_positive_pairs:
            train_interaction_matrix[drug_idx, target_idx] = 0.0

        print("Computing fold-local LapRLS scores...")
        lap_scores = LapRLS(drug_sim_1, drug_sim_2, target_sim_1, target_sim_2).predict(train_interaction_matrix)

        train_unknown_pairs = [
            (int(drug_idx), int(target_idx))
            for drug_idx in train_drug_indices
            for target_idx in train_target_indices
            if interaction_matrix[int(drug_idx), int(target_idx)] == 0 and (int(drug_idx), int(target_idx)) not in warm_start_test_positive_pairs
        ]
        negative_count = min(len(train_unknown_pairs), len(train_positive_pairs) * config.neg_ratio)
        train_negative_pairs = sample_low_score_negatives(train_unknown_pairs, lap_scores, negative_count)
        if len(train_negative_pairs) == 0 and len(train_unknown_pairs) > 0:
            train_negative_pairs = random.sample(train_unknown_pairs, k=min(len(train_unknown_pairs), max(1, len(train_positive_pairs))))

        print(f"Training samples: positive={len(train_positive_pairs)}, pseudo-negative={len(train_negative_pairs)}")
        samples = [(drug_idx, target_idx, 1.0) for drug_idx, target_idx in train_positive_pairs]
        samples += [(drug_idx, target_idx, 0.0) for drug_idx, target_idx in train_negative_pairs]
        if len(samples) == 0:
            print("No training samples were available in this fold; the fold was skipped.")
            continue

        train_drug_map = {int(global_idx): local_idx for local_idx, global_idx in enumerate(train_drug_indices)}
        train_target_map = {int(global_idx): local_idx for local_idx, global_idx in enumerate(train_target_indices)}

        train_drug_similarity = drug_similarity[np.ix_(train_drug_indices, train_drug_indices)]
        train_target_similarity = target_similarity[np.ix_(train_target_indices, train_target_indices)]
        train_drug_edges = build_knn_graph(train_drug_similarity, config.knn_k, device)
        train_target_edges = build_knn_graph(train_target_similarity, config.knn_k, device)
        train_drug_features = drug_features[train_drug_indices]
        train_target_features = target_features[train_target_indices]

        def collate_fn(batch):
            return (
                torch.tensor([train_drug_map[item[0]] for item in batch], dtype=torch.long),
                torch.tensor([train_target_map[item[1]] for item in batch], dtype=torch.long),
                torch.tensor([item[2] for item in batch], dtype=torch.float32),
            )

        loader = torch.utils.data.DataLoader(samples, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

        model = LNMGAT(n_drugs, n_targets, config.hidden_dim, config.heads, config.dropout).to(device)
        model.load_state_dict(base_state_dict)
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        epoch_bar = tqdm(range(config.epochs), desc=f"Fold {fold}", leave=False)
        for _epoch in epoch_bar:
            total_loss = 0.0
            for drug_batch, target_batch, label_batch in loader:
                drug_batch = drug_batch.to(device)
                target_batch = target_batch.to(device)
                label_batch = label_batch.to(device)
                optimizer.zero_grad()
                logits = model(train_drug_features, train_drug_edges, train_target_features, train_target_edges, drug_batch, target_batch)
                loss = criterion(logits, label_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                total_loss += float(loss.item())
            scheduler.step()
            epoch_bar.set_postfix({"loss": f"{total_loss / max(1, len(loader)):.4f}"})

        model.eval()
        gat_scores = model.predict_full(drug_features, full_drug_edges, target_features, full_target_edges, batch_size=config.predict_batch_size)

        metrics, prediction_rows = evaluate_scenarios(
            fold=fold,
            interaction_matrix=interaction_matrix,
            train_drug_indices=train_drug_indices,
            test_drug_indices=test_drug_indices,
            train_target_indices=train_target_indices,
            test_target_indices=test_target_indices,
            train_positive_pairs=train_positive_pairs,
            warm_start_test_positive_pairs=warm_start_test_positive_pairs,
            gat_scores=gat_scores,
            lap_scores=lap_scores,
            config=config,
        )
        all_metrics.append(metrics)
        all_prediction_rows.extend(prediction_rows)

        print("Fold evaluation completed. Model weights are not saved in this no-checkpoint release.")

    write_metric_summaries(all_metrics, args.output_dir, args.dataset)
    save_prediction_tables(all_prediction_rows, drug_ids, target_ids, args.output_dir, args.dataset, prefix="LNMGAT")

    config_path = os.path.join(args.output_dir, f"LNMGAT_Config_{args.dataset}.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, indent=2)
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
