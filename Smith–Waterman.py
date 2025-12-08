import pandas as pd
import numpy as np
from Bio import Align

# ==============================
# Parameter settings
# ==============================
input_file = "./data/davis_target.xlsx"
output_file = "similarity_matrix.csv"
seq_column = "target_sequence"
id_column = "targetID"

# ==============================
# 1. Read data
# ==============================
df = pd.read_excel(input_file)

if id_column not in df.columns:
    df[id_column] = [f"Target_{i+1}" for i in range(len(df))]

if seq_column not in df.columns:
    raise ValueError(f"Column '{seq_column}' not found. Please check the Excel file column names.")

targets = df[id_column].tolist()
sequences = df[seq_column].tolist()
n = len(sequences)

# ==============================
# 2. Initialize aligner
# ==============================
aligner = Align.PairwiseAligner()
aligner.mode = "local"        # Smith–Waterman algorithm
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.open_gap_score = -0.5
aligner.extend_gap_score = -0.1

# ==============================
# 3. Compute raw score matrix
# ==============================
matrix = np.zeros((n, n))
self_scores = np.zeros(n)

for i in range(n):
    self_scores[i] = aligner.score(sequences[i], sequences[i])  # Self-alignment score

for i in range(n):
    for j in range(i, n):
        score_ij = (aligner.score(sequences[i], sequences[j]) +
                    aligner.score(sequences[j], sequences[i])) / 2
        matrix[i, j] = score_ij
        matrix[j, i] = score_ij  # Ensure symmetry

# ==============================
# 4. Normalize to [0, 1]
# ==============================
norm_matrix = np.zeros_like(matrix)
for i in range(n):
    for j in range(n):
        denom = np.sqrt(self_scores[i] * self_scores[j])
        norm_matrix[i, j] = matrix[i, j] / denom if denom > 0 else 0

# ==============================
# 5. Save results
# ==============================
sim_df = pd.DataFrame(norm_matrix, index=targets, columns=targets)
sim_df = sim_df.round(5)
sim_df.to_csv(output_file, float_format="%.5f")

print(f"✅ The similarity matrix has been saved as '{output_file}', "
      f"values are normalized to [0, 1], and symmetry is ensured.")
