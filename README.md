# LNMGAT Reproducibility Package

This repository provides the source code required to reproduce the LNMGAT drug-target interaction prediction experiments from the input matrices. 
## Files

| File | Purpose |
| --- | --- |
| `lnmgat_model.py` | Reusable implementation of LapRLS, the dual-branch graph attention model, data loading utilities, evaluation metrics, and prediction-table export. |
| `train.py` | Trains and evaluates LNMGAT under warm-start, new-drug, new-target, and new-both settings. It exports metrics and prediction tables only. |
| `test.py` | Testing entry point that reruns the same deterministic evaluation protocol from scratch. It does not require checkpoint files. |
| `calculate_drug_similarity.py` | Builds a drug similarity matrix from SMILES strings using Morgan fingerprints and Tanimoto similarity. |
| `calculate_target_similarity.py` | Builds a target similarity matrix from protein sequences using Smith-Waterman local alignment. |
| `requirements.txt` | Python package requirements. |
| `REVIEWER_RESPONSE_CODE_AVAILABILITY.md` | Suggested response to the reviewer and revised Code Availability wording. |

## Expected data layout

Place the benchmark matrices under `./data_DTI` by default:

```text
data_DTI/
  drug_similarity_davis.csv
  drug_IS_0_davis.csv
  target_similarity_davis.csv
  target_IS_0_davis.csv
  interaction_0_davis.csv
```

The default filenames follow this pattern:

```text
drug_similarity_<dataset>.csv
drug_IS_<label>_<dataset>.csv
target_similarity_<dataset>.csv
target_IS_<label>_<dataset>.csv
interaction_<label>_<dataset>.csv
```

Each matrix should be a CSV file with row identifiers in the first column and column identifiers in the header. The interaction matrix must have shape `[n_drugs, n_targets]`, with known positive interactions encoded as `1` and unknown pairs encoded as `0`.

## Environment setup

```bash
conda create -n lnmgat python=3.10 -y
conda activate lnmgat
pip install -r requirements.txt
```

PyTorch Geometric must match the installed PyTorch and CUDA versions. Follow the official PyTorch Geometric installation command for your environment if the generic installation fails.

## Optional similarity construction

Drug similarity from SMILES:

```bash
python calculate_drug_similarity.py
```

Target similarity from protein sequences:

```bash
python calculate_target_similarity.py
```

Adjust the input file paths and column names inside these scripts if your raw Excel files use different names.

## Train and evaluate from scratch

```bash
python train.py \
  --dataset davis \
  --label 0 \
  --data-dir ./data_DTI \
  --output-dir ./output \
  --epochs 300 \
  --n-splits 5 \
  --seed 42
```

The script writes:

```text
output/LNMGAT_Fold_Metrics_davis.csv
output/LNMGAT_Mean_Metrics_davis.csv
output/LNMGAT_Predictions_davis_Warm-Start.csv
output/LNMGAT_Predictions_davis_New-Drug.csv
output/LNMGAT_Predictions_davis_New-Target.csv
output/LNMGAT_Predictions_davis_New-Both.csv
output/LNMGAT_Config_davis.json
```



## Testing entry point

`test.py` is provided for reviewers who expect a separate testing script. In this no-checkpoint release, it reruns the same deterministic cross-validation and evaluation protocol from scratch:

```bash
python test.py \
  --dataset davis \
  --label 0 \
  --data-dir ./data_DTI \
  --output-dir ./reproduced_results \
  --epochs 300 \
  --n-splits 5 \
  --seed 42
```

This produces the same output file types as `train.py`, under the selected output directory.

## Running other datasets

Use the corresponding dataset name and input files:

```bash
python train.py --dataset kiba --label 0 --data-dir ./data_DTI --output-dir ./output_kiba
python train.py --dataset bindingdb --label 0 --data-dir ./data_DTI --output-dir ./output_bindingdb
```

If your filenames differ from the default pattern, pass explicit filenames:

```bash
python train.py \
  --dataset davis \
  --label 0 \
  --data-dir ./data_DTI \
  --drug-sim-file-1 custom_drug_similarity.csv \
  --drug-sim-file-2 custom_drug_integrated_similarity.csv \
  --target-sim-file-1 custom_target_similarity.csv \
  --target-sim-file-2 custom_target_integrated_similarity.csv \
  --interaction-file custom_interaction.csv
```

## Reproducibility notes

1. The random seed is controlled by `--seed` and is also used for the drug, target, and positive-pair split generators.
2. LapRLS scores are computed fold locally after masking the held-out regions, which avoids using held-out labels during pseudo-negative mining.
3. Pseudo-negative pairs are selected from unknown pairs with the lowest fold-local LapRLS scores.
4. Warm-start, new-drug, new-target, and new-both evaluations are all produced within the same run.