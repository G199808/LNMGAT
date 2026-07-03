"""
Run the LNMGAT testing protocol from scratch without loading checkpoint files.

This repository version does not distribute, save, or load model checkpoint files.
The testing entry point reuses the deterministic cross-validation and evaluation
pipeline implemented in train.py, so reviewers can reproduce the reported tables
from the input matrices directly.

Example:
    python test.py --dataset davis --label 0 --data-dir ./data_DTI \
        --output-dir ./reproduced_results --epochs 300
"""

from train import main


if __name__ == "__main__":
    main()
