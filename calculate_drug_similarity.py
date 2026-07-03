"""
Compute a drug-drug similarity matrix from SMILES strings using RDKit fingerprints.

The default input follows the Davis example used in the manuscript:
    ./data/davis_drug.xlsx

Example:
    python calculate_drug_similarity.py \
        --input-file ./data/davis_drug.xlsx \
        --output-file drug_similarity_davis.csv \
        --id-column compound_id \
        --smiles-column compound_iso_smiles
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RDKit Morgan fingerprint Tanimoto drug similarities.")
    parser.add_argument("--input-file", type=str, default="./data/davis_drug.xlsx", help="Input Excel file containing drug identifiers and SMILES strings.")
    parser.add_argument("--output-file", type=str, default="drug_similarity_matrix_rdkit.csv", help="Output CSV similarity matrix.")
    parser.add_argument("--sheet-name", type=str, default="Sheet1", help="Excel sheet name.")
    parser.add_argument("--id-column", type=str, default="compound_id", help="Drug identifier column.")
    parser.add_argument("--smiles-column", type=str, default="compound_iso_smiles", help="SMILES column.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument("--n-bits", type=int, default=1024, help="Morgan fingerprint bit length.")
    parser.add_argument("--digits", type=int, default=5, help="Number of decimals in the output matrix.")
    return parser.parse_args()


def calculate_fingerprint_similarity(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 1024) -> float:
    """Return Morgan fingerprint Tanimoto similarity for two SMILES strings."""
    try:
        mol_a = Chem.MolFromSmiles(str(smiles_a))
        mol_b = Chem.MolFromSmiles(str(smiles_b))
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return 0.0


def main() -> None:
    args = parse_args()
    frame = pd.read_excel(args.input_file, sheet_name=args.sheet_name)
    required_columns = {args.id_column, args.smiles_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    drug_ids = frame[args.id_column].astype(str).tolist()
    smiles_list = frame[args.smiles_column].astype(str).tolist()
    n_drugs = len(drug_ids)
    similarity_matrix = np.zeros((n_drugs, n_drugs), dtype=np.float32)

    for i in range(n_drugs):
        for j in range(i, n_drugs):
            if i == j:
                similarity = 1.0
            else:
                similarity = calculate_fingerprint_similarity(smiles_list[i], smiles_list[j], args.radius, args.n_bits)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    similarity_matrix = np.round(similarity_matrix, args.digits)
    similarity_frame = pd.DataFrame(similarity_matrix, index=drug_ids, columns=drug_ids)
    similarity_frame.to_csv(args.output_file, float_format=f"%.{args.digits}f")
    print(f"Drug similarity matrix saved to: {args.output_file}")


if __name__ == "__main__":
    main()
