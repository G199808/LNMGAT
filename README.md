# LNMGAT
**L**aplacian Regularized **N**egative-Mining Graph **A**ttention Ne**t**work  

**LNMGAT: A Laplacian Regularized Negative-Mining Graph Attention Network for Robust Drug–Target Interaction Prediction Under Multi-Scenario Cold-Start Settings**  
Shuai Guo, Weichi Liu, Jie Zou, Tao Ban, Gaifang Dong*  

## Key Features
- **Zero experimentally verified negative samples** are used throughout training and testing  
- LapRLS-guided reliable negative mining from unlabeled drug–target pairs  
- Multi-head Graph Attention Network (GAT) on heterogeneous drug/target similarity graphs  
- Strict 5-fold cross-validation with four realistic scenarios:  
  Warm-Start • New Drug (cold-drug) • New Target (cold-target) • New Both (cold-pair)  
- State-of-the-art AUPR/AUROC on Yamanishi, Davis, KIBA, and BindingDB  

## Quick Start

### 1. Environment
```bash
conda create -n lnmgat python=3.9 -y
conda activate lnmgat
git clone https://github.com/yourname/LNMGAT.git
cd LNMGAT
pip install -r requirements.txt
