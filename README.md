# SoftMoE-SHAP-MOOC-Dropout-Prediction

Official implementation of:

"SoftMoE-SHAP: A Calibrated Shared-Trunk Soft Mixture-of-Experts With SHAP-Anchored Gate Priors for MOOC Dropout Prediction"

## Overview

This repository contains the implementation of the proposed SoftMoE-SHAP framework for MOOC dropout prediction.

Included components:

- Shared-trunk Soft Mixture-of-Experts model
- SHAP-guided gate prior
- TabNet baseline
- Calibration analysis
- Ablation experiments
- Statistical evaluation

## Dataset

Experiments are based on the KDD Cup 2015 / XuetangX MOOC dataset.

Raw data are not redistributed due to dataset availability restrictions.

## Supplementary Material

The repository includes the complete feature dictionary for the 490 engineered behavioral features.

## Reproducibility

Install dependencies:

pip install -r requirements.txt

Run:

python src/run_soft_moe_shap.py
