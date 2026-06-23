# LNMGAT: Laplacian Regularized Pseudo-Negative Mining Graph Attention Network

This repository provides the source code, processed datasets, running scripts, pretrained checkpoints, and reproducibility instructions for the paper:

**LNMGAT: A Laplacian Regularized Pseudo-Negative Mining Graph Attention Network for Robust Drug–Target Interaction Prediction Under Multi-Scenario Cold-Start Settings**

LNMGAT is a drug–target interaction prediction framework that integrates LapRLS-guided reliable pseudo-negative mining with dual graph attention encoders. The model is evaluated under warm-start, drug cold-start, target cold-start, and pair cold-start settings.

---

## 1. Repository Structure

```text
LNMGAT/
├── data_DTI/
│   ├── yamanishi/
│   ├── davis/
│   ├── kiba/
│   └── bindingdb/
├── checkpoints/
│   ├── yamanishi/
│   ├── davis/
│   ├── kiba/
│   └── bindingdb/
├── scripts/
│   ├── preprocess/
│   ├── train/
│   ├── test/
│   └── reproduce_tables/
├── output/
├── main.py
├── model.py
├── train.py
├── test.py
├── reproduce_tables.py
├── requirements.txt
├── environment.yml
└── README.md
The main folders are organized as follows:
·data_DTI/: processed benchmark datasets and similarity matrices.
·checkpoints/: pretrained model checkpoints for each benchmark dataset.
·scripts/preprocess/: scripts for dataset preparation and preprocessing.
·scripts/train/: scripts for training LNMGAT and reproduced baselines.
·scripts/test/: scripts for testing pretrained models.
·scripts/reproduce_tables/: scripts for reproducing the tables reported in the manuscript.
·output/: generated fold-level metrics, mean performance summaries, and prediction files.

2. Environment Setup
We recommend using Conda to create an isolated environment.
conda create -n lnmgat python=3.9
conda activate lnmgat
Install dependencies using:
pip install -r requirements.txt
Alternatively, the environment can be created using:
conda env create -f environment.yml
conda activate lnmgat
Recommended dependency versions:
python==3.9
numpy==1.24.4
pandas==2.0.3
scipy==1.10.1
scikit-learn==1.3.0
torch==2.0.1
torch-geometric==2.3.1
tqdm==4.66.1
matplotlib==3.7.2
Please install the PyTorch and PyTorch Geometric versions compatible with your CUDA version. For CPU-only execution, install the CPU versions of PyTorch and PyTorch Geometric.

3. Dataset Preparation
The processed benchmark datasets are provided under:
data_DTI/
Each dataset folder contains the interaction matrix, drug similarity matrices, target similarity matrices, and interaction-sharing similarity matrices.
Expected file examples:
data_DTI/davis/
├── interaction_0_davis.csv
├── drug_similarity_davis.csv
├── drug_IS_0_davis.csv
├── target_similarity_davis.csv
└── target_IS_0_davis.csv
If the processed datasets are already provided, no additional preprocessing is required.
To regenerate processed files from raw data, run:
python scripts/preprocess/prepare_yamanishi.py
python scripts/preprocess/prepare_davis.py
python scripts/preprocess/prepare_kiba.py
python scripts/preprocess/prepare_bindingdb.py

4. Pretrained Checkpoints
Pretrained checkpoints are organized by dataset under:
checkpoints/
Expected checkpoint structure:
checkpoints/
├── yamanishi/
│   ├── fold1.pt
│   ├── fold2.pt
│   ├── fold3.pt
│   ├── fold4.pt
│   ├── fold5.pt
│   └── config.json
├── davis/
│   ├── fold1.pt
│   ├── fold2.pt
│   ├── fold3.pt
│   ├── fold4.pt
│   ├── fold5.pt
│   └── config.json
├── kiba/
│   ├── fold1.pt
│   ├── fold2.pt
│   ├── fold3.pt
│   ├── fold4.pt
│   ├── fold5.pt
│   └── config.json
└── bindingdb/
    ├── fold1.pt
    ├── fold2.pt
    ├── fold3.pt
    ├── fold4.pt
    ├── fold5.pt
    └── config.json
If checkpoint files exceed the GitHub file-size limit, they are provided through the GitHub Release page or an external archival link. Please download the checkpoint files and place them in the corresponding dataset folder under checkpoints/.
Example:
mkdir -p checkpoints/davis
# Download fold1.pt to fold5.pt and config.json
# Place them under checkpoints/davis/

5. Training LNMGAT
To train LNMGAT on each benchmark dataset, run:
python train.py --dataset yamanishi --seed 42 --n_splits 5
python train.py --dataset davis --seed 42 --n_splits 5
python train.py --dataset kiba --seed 42 --n_splits 5
python train.py --dataset bindingdb --seed 42 --n_splits 5
Main arguments:
--dataset       Dataset name: yamanishi, davis, kiba, or bindingdb
--seed          Random seed
--n_splits      Number of cross-validation folds
--epochs        Number of training epochs
--batch_size    Batch size
--hidden_dim    Hidden dimension
--heads         Number of GAT attention heads
--knn_k         Number of k-nearest neighbors
--neg_ratio     Training pseudo-negative ratio
Example:
python train.py --dataset davis --seed 42 --n_splits 5 --epochs 300 --batch_size 512
Training outputs will be saved under:
output/
checkpoints/

6. Testing Pretrained Models
To evaluate pretrained checkpoints, run:
python test.py --dataset yamanishi --checkpoint_dir checkpoints/yamanishi/
python test.py --dataset davis --checkpoint_dir checkpoints/davis/
python test.py --dataset kiba --checkpoint_dir checkpoints/kiba/
python test.py --dataset bindingdb --checkpoint_dir checkpoints/bindingdb/
The test script generates fold-level metrics and summary results under:
output/
Example output files:
output/LNMGAT_Fold_Metrics_davis.csv
output/LNMGAT_Mean_Metrics_davis.csv
output/LNMGAT_Predictions_davis_Warm-Start.csv
output/LNMGAT_Predictions_davis_New-Drug.csv
output/LNMGAT_Predictions_davis_New-Target.csv
output/LNMGAT_Predictions_davis_New-Both.csv

7. Reproducing Main Tables
The following commands reproduce the main performance tables reported in the manuscript.
Table 4: Yamanishi dataset
python reproduce_tables.py --table 4 --dataset yamanishi
Table 5: Davis dataset
python reproduce_tables.py --table 5 --dataset davis
Table 6: KIBA dataset
python reproduce_tables.py --table 6 --dataset kiba
Table 7: BindingDB dataset
python reproduce_tables.py --table 7 --dataset bindingdb
Table 9: Ablation study on Davis
python reproduce_tables.py --table 9 --dataset davis --ablation
The reproduced tables will be saved under:
output/tables/

8. Reproducing Ablation Experiments
The ablation study compares the full LNMGAT model with two representative variants:
·Full LNMGAT: full model with LapRLS-guided pseudo-negative mining and dual GAT encoders.
·LapRLS_Only: prediction using only the LapRLS-based similarity propagation module.
·GAT_Only: GAT model using random pseudo-negative sampling instead of LapRLS-guided pseudo-negative mining.
Run:
python scripts/reproduce_tables/reproduce_table9_ablation.py --dataset davis --seed 42 --n_splits 5
Or:
python reproduce_tables.py --table 9 --dataset davis --ablation

9. Reproducing Baseline Comparisons
All baseline methods reported in the manuscript were reproduced under the same preprocessing, data partitions, cold-start scenario definitions, and pseudo-negative sampling protocol.
To reproduce baseline results, run:
python scripts/train/train_baselines.py --dataset yamanishi --n_splits 5
python scripts/train/train_baselines.py --dataset davis --n_splits 5
python scripts/train/train_baselines.py --dataset kiba --n_splits 5
python scripts/train/train_baselines.py --dataset bindingdb --n_splits 5
To summarize baseline and LNMGAT results:
python scripts/reproduce_tables/summarize_main_results.py

10. Evaluation Scenarios
LNMGAT is evaluated under four prediction scenarios:
1.Warm Start
Both the drug and the target are observed in the supervised training pairs.
2.Drug Cold Start
The drug is absent from the supervised training pairs, while the target is observed.
3.Target Cold Start
The target is absent from the supervised training pairs, while the drug is observed.
4.Pair Cold Start
Neither the drug nor the target is observed in the supervised training pairs.
For each fold, test interaction labels are masked before LapRLS-based pseudo-negative mining. The drug-drug and target-target similarity matrices are used as fixed side information.

11. Output Metrics
The following metrics are reported:
AUPR
AUROC
F1-score
Precision
Recall
The main tables report mean and standard deviation over 5-fold cross-validation.

12. Statistical Significance
For comparisons with reproduced baseline methods, statistical significance is assessed using fold-level performance values. LNMGAT is compared with the strongest competing baseline in each setting.
Significance markers:
*  p < 0.05
** p < 0.01
The exact statistical test used in the manuscript is specified in the table notes.

13. Random Seeds and Reproducibility
The default random seed is:
42
To improve reproducibility, the code fixes random seeds for:
·Python random
·NumPy
·PyTorch
·CUDA, when available
Example:
python train.py --dataset davis --seed 42 --n_splits 5

14. Hardware
Experiments can be run on either CPU or GPU. GPU acceleration is recommended.
The default device is automatically selected:
cuda if torch.cuda.is_available() else cpu
Recommended configuration:
GPU: NVIDIA GPU with at least 8 GB memory
CPU: 8 cores or above
RAM: 32 GB or above
For large datasets such as BindingDB, more GPU memory and system RAM are recommended.

15. Citation
If you use this repository, please cite:
Guo S, Liu W, Zou J, Ban T, Dong G.
LNMGAT: A Laplacian Regularized Pseudo-Negative Mining Graph Attention Network
for Robust Drug-Target Interaction Prediction Under Multi-Scenario Cold-Start Settings.

16. Contact
For questions, please contact:
Gaifang Dong
College of Computer and Information Engineering
Inner Mongolia Agricultural University
Email: donggf@imau.edu.cn

17. Notes
·The processed datasets are provided for reproducibility.
·Pretrained checkpoints are stored under checkpoints/ or provided through the repository release page.
·Fold-level results and summary tables are saved under output/.
·The docking analysis in the manuscript provides auxiliary in silico support and should not be interpreted as wet-lab validation.

