# AFS-GAN + BiLSTM-Transformer + Meta-Learner for Software Defect Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1.0-orange)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official implementation of the paper:

> **AFS-GAN + BiLSTM-Transformer: A Novel Adaptive Feature Selection and
> Generative Adversarial Network-Augmented Bidirectional LSTM-Transformer
> Framework for Software Defect Prediction**
>
> Bhuvaneshwaran A, Deisy C — KGISL Institute of Technology / TCE, 2025

---

## Architecture

```
Raw input X (n modules x d features)
    │
    ▼
Phase 1 — Preprocessing
    Median imputation → IQR outlier clipping → Z-score normalization
    │
    ▼
Phase 2a — AFS-SHAP Feature Selection
    Joint score: rho_j = alpha * MI(X_j; Y) + (1-alpha) * |SHAP_j|
    Greedy K* selection with PAC-learning bound
    │
Phase 2b — DC-GAN Augmentation
    Conditional WGAN-GP → synthesizes minority class to 1:1 ratio
    │
    ▼ augmented D_aug
Phase 3 — BiLSTM-Transformer Dual Encoder
    BiLSTM (H=256x2)  ←──────────────────────┐
    Transformer (d=512, 8 heads, 4 layers)    │
    Cross-attention: Z queries H              │
    Mean pool → z_final (512-dim)             │
    │                                         │
    ▼                                         │
Phase 4 — Stacked Ensemble Meta-Learner       │
    BiLSTM-T probs + XGBoost probs + RF probs │
    Logistic regression meta-model            │
    Youden's J threshold                      │
    │
    ▼
P(defect | x) → binary label
```

---

## Results (NASA PROMISE, 10-fold CV)

| Dataset | Accuracy | F1-Score | AUC-ROC | MCC | FPR | FNR |
|---------|----------|----------|---------|-----|-----|-----|
| CM1 | 0.921 | 0.893 | 0.951 | 0.841 | 0.079 | 0.093 |
| JM1 | 0.934 | 0.911 | 0.963 | 0.867 | 0.066 | 0.074 |
| KC1 | 0.947 | 0.928 | 0.974 | 0.891 | 0.053 | 0.062 |
| PC1 | 0.958 | 0.941 | 0.981 | 0.908 | 0.042 | 0.051 |
| MC2 | 0.939 | 0.918 | 0.967 | 0.876 | 0.061 | 0.071 |
| **Avg** | **0.940** | **0.918** | **0.967** | **0.877** | **0.060** | **0.070** |

---

## Setup

```bash
git clone https://github.com/bhuvanesh0312/afs-gan-bilstm-transformer.git
cd afs-gan-bilstm-transformer
pip install -r requirements.txt
```

### Download Datasets

Download NASA PROMISE datasets (.arff) from:
https://promise.site.uottawa.ca/SERepository/

Place in `data/raw/`:
```
data/raw/cm1.arff
data/raw/jm1.arff
data/raw/kc1.arff
data/raw/pc1.arff
data/raw/mc2.arff
```

---

## Run

```bash
# Single dataset
python run_experiment.py --dataset KC1 --data_path data/raw/kc1.arff

# GPU
python run_experiment.py --dataset JM1 --data_path data/raw/jm1.arff --device cuda

# Smoke test (no real data needed)
python run_experiment.py --dataset KC1 --synthetic

# All 5 datasets
bash run_all_datasets.sh
# or with GPU:
bash run_all_datasets.sh --device cuda
```

---

## Project Structure

```
afs_gan_bilstm/
├── configs/
│   └── default.yaml          # All hyperparameters
├── data/
│   ├── loader.py             # PROMISE .arff loader + synthetic fallback
│   ├── preprocessing.py      # Phase 1: impute, clip, normalize
│   ├── afs_shap.py           # Phase 2a: AFS-SHAP feature selection
│   ├── dcgan.py              # Phase 2b: DC-GAN (cWGAN-GP) augmentation
│   └── sequencer.py          # Tabular → version sequence (T x K)
├── models/
│   ├── bilstm_transformer.py # Phase 3: BiLSTM + Transformer + cross-attention
│   ├── deep_head.py          # Full deep classifier
│   └── meta_learner.py       # Phase 4: XGBoost + RF + LogReg stacking
├── training/
│   └── trainer.py            # Training loop, early stopping, Youden threshold
├── evaluation/
│   ├── metrics.py            # All metrics, fold aggregation, Wilcoxon, Cohen's d
│   └── visualize.py          # ROC, CM, attention heatmap, feature importance
├── utils/
│   ├── config.py             # YAML config loader
│   └── logger.py             # Experiment logger (JSON output)
├── tests/
│   └── test_smoke.py         # Smoke tests (pytest)
├── run_experiment.py         # Main entry point
├── run_all_datasets.sh       # Run all 5 datasets
└── requirements.txt
```

---

## Key Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| AFS-SHAP | alpha (MI weight) | 0.6 |
| DC-GAN | GP penalty lambda | 10.0 |
| DC-GAN | Latent dim k | 100 |
| BiLSTM | Hidden units H | 256 (x2 bidirectional) |
| Transformer | d_model | 512 |
| Transformer | Attention heads | 8 |
| Transformer | Encoder layers | 4 |
| Training | Learning rate | 3e-4 |
| Training | Max epochs | 200 (early stop patience=15) |
| Ensemble | CV folds (stacking) | 10 |

---

## Tests

```bash
python -m pytest tests/test_smoke.py -v
```

---

## Citation

```bibtex
@article{bhuvaneshwaran2025afsgan,
  title     = {AFS-GAN + BiLSTM-Transformer: A Novel Framework for Software Defect Prediction},
  author    = {Bhuvaneshwaran, A and Deisy, C},
  journal   = {Information and Software Technology},
  year      = {2025},
  publisher = {Elsevier}
}
```

---

## Contact

- Bhuvaneshwaran A — bhuvanesh0312@gmail.com (KGISL Institute of Technology)
- Deisy C — cdcse@tce.edu (Thiagarajar College of Engineering)
