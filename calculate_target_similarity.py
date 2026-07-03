"""
Compute a target-target sequence similarity matrix using Smith-Waterman local alignment.

Example:
    python calculate_target_similarity.py \
        --input-file ./data/davis_target.xlsx \
        --output-file target_similarity_davis.csv \
        --id-column targetID \
        --sequence-column target_sequence
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from Bio import Align


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute normalized Smith-Waterman target sequence similarities.")
    parser.add_argument("--input-file", type=str, default="./data/davis_target.xlsx", help="Input Excel file containing target identifiers and protein sequences.")
    parser.add_argument("--output-file", type=str, default="similarity_matrix.csv", help="Output CSV similarity matrix.")
    parser.add_argument("--id-column", type=str, default="targetID", help="Target identifier column.")
    parser.add_argument("--sequence-column", type=str, default="target_sequence", help="Protein sequence column.")
    parser.add_argument("--match-score", type=float, default=2.0, help="Alignment match score.")
    parser.add_argument("--mismatch-score", type=float, default=-1.0, help="Alignment mismatch score.")
    parser.add_argument("--open-gap-score", type=float, default=-0.5, help="Gap opening penalty.")
    parser.add_argument("--extend-gap-score", type=float, default=-0.1, help="Gap extension penalty.")
    parser.add_argument("--digits", type=int, default=5, help="Number of decimals in the output matrix.")
    return parser.parse_args()


def build_aligner(args: argparse.Namespace) -> Align.PairwiseAligner:
    """Create a Smith-Waterman local alignment object."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = args.match_score
    aligner.mismatch_score = args.mismatch_score
    aligner.open_gap_score = args.open_gap_score
    aligner.extend_gap_score = args.extend_gap_score
    return aligner


def main() -> None:
    args = parse_args()
    frame = pd.read_excel(args.input_file)

    if args.id_column not in frame.columns:
        frame[args.id_column] = [f"Target_{index + 1}" for index in range(len(frame))]
    if args.sequence_column not in frame.columns:
        raise ValueError(f"Required sequence column was not found: {args.sequence_column}")

    target_ids = frame[args.id_column].astype(str).tolist()
    sequences = frame[args.sequence_column].astype(str).tolist()
    n_targets = len(sequences)
    aligner = build_aligner(args)

    raw_scores = np.zeros((n_targets, n_targets), dtype=np.float32)
    self_scores = np.zeros(n_targets, dtype=np.float32)

    for i in range(n_targets):
        self_scores[i] = float(aligner.score(sequences[i], sequences[i]))

    for i in range(n_targets):
        for j in range(i, n_targets):
            score_ij = (float(aligner.score(sequences[i], sequences[j])) + float(aligner.score(sequences[j], sequences[i]))) / 2.0
            raw_scores[i, j] = score_ij
            raw_scores[j, i] = score_ij

    normalized_scores = np.zeros_like(raw_scores)
    for i in range(n_targets):
        for j in range(n_targets):
            denominator = np.sqrt(self_scores[i] * self_scores[j])
            normalized_scores[i, j] = raw_scores[i, j] / denominator if denominator > 0 else 0.0

    similarity_frame = pd.DataFrame(np.round(normalized_scores, args.digits), index=target_ids, columns=target_ids)
    similarity_frame.to_csv(args.output_file, float_format=f"%.{args.digits}f")
    print(f"Target similarity matrix saved to: {args.output_file}")


if __name__ == "__main__":
    main()
