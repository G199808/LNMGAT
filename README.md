# LNMGAT
**L**aplacian Regularized **N**egative-Mining Graph **A**ttention Ne**t**work  

Official PyTorch + PyTorch-Geometric implementation of the paper:  

**LNMGAT: A Laplacian Regularized Negative-Mining Graph Attention Network for Robust Drug–Target Interaction Prediction Under Multi-Scenario Cold-Start Settings**  
Shuai Guo, Weichi Liu, Jie Zou, Tao Ban, Gaifang Dong*  
*Inner Mongolia Agricultural University, 2025 (Under Review)*  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

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
