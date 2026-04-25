# Statistical Arbitrage on the S&P 500
### Extending Krauss, Do & Huck (2016) with LSTM and CNN Architectures
*IEOR 4733: Algorithmic Trading — Course Project (Option A: Reproduce & Extend)*

---

## Overview

This project reproduces and extends the statistical arbitrage framework from **Krauss, Do & Huck (2016)** — *"Deep Neural Networks, Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500"*. The original paper trains a DNN, GBT, Random Forest, and an equal-weighted ensemble on 31 hand-crafted lagged return features to forecast which S&P 500 stocks will outperform the cross-sectional median the following day. We extend this by introducing two new architectures — an **LSTM** and a **1D CNN** — that operate directly on raw return sequences rather than pre-engineered features.

---

## The Original Paper

### Setup
- **Universe:** S&P 500 constituents (survivor-bias corrected), 1992–2015
- **Data source:** CRSP via WRDS — daily total return indices (`ret` from `crsp.dsf`)
- **Input features:** 31 lagged returns per stock per day — R(1)…R(20) at daily resolution, then R(40), R(60)…R(240) at monthly resolution
- **Target variable:** Binary — does stock *s* outperform the cross-sectional median return on day *t+1*?
- **Training regime:** Sliding window — 750-day training set, 250-day test set, 23 non-overlapping batches
- **Trading logic:** Rank all stocks by P(outperform); long top-k, short bottom-k

### Key Results
| Model | Daily Return (pre-cost) | Daily Return (post-cost) | Sharpe Ratio |
|-------|------------------------|--------------------------|--------------|
| DNN   | 0.33%                  | 0.13%                    | 0.55         |
| GBT   | 0.37%                  | 0.17%                    | 1.23         |
| RAF   | 0.43%                  | 0.23%                    | 1.90         |
| ENS   | 0.45%                  | 0.25%                    | 1.81         |

### Key Finding on Variable Importance
The paper's variable importance analysis revealed that **R(1)–R(5) — the past 5 trading days — carry the highest predictive power** across all three base models. Returns in the R(10)–R(20) range rank lowest. This finding directly motivates the CNN extension.

---

## Our Extensions

### Why Not Just Use the 31 Features?

The original paper's 31 features are a hand-crafted, multi-resolution encoding of the return history. While effective, this representation makes two implicit assumptions:

1. The chosen lag intervals (daily for 20 days, then monthly) are the right resolution
2. Temporal ordering and sequential dependencies within the series are not important beyond what these aggregated features capture

Our extensions challenge both assumptions by feeding **raw return sequences directly** into architectures whose inductive biases are designed for sequential data.

---

### Extension 1: LSTM

**Input:** Raw daily return sequence of length `SEQUENCE_LENGTH` (default 240), shape `(n_samples, SEQUENCE_LENGTH)` — reshaped internally to `(n_samples, SEQUENCE_LENGTH, 1)` for PyTorch.

**Architecture:**
```
Input (batch, seq_len, 1)
        ↓
Stacked LSTM (n_layers, hidden_size)
        ↓
Final hidden state h_T  (batch, hidden_size)
        ↓
Dropout
        ↓
Linear (hidden_size → 1)
        ↓
Sigmoid → P(outperform cross-sectional median)
```

**Loss function:** Binary cross-entropy

**Hypothesis:**
> Hand-crafted lag features cannot fully capture the non-linear sequential dependencies present in raw return history. An LSTM maintaining hidden state across the full sequence can learn which temporal patterns are predictive without requiring pre-specified lag intervals — and should outperform the 31-feature DNN baseline.

**What this tests:** Whether **long-range sequential memory and learned temporal feature extraction** outperform fixed, hand-engineered lag aggregation.

**Key implementation details:**
- Gradient clipping (`max_norm=1.0`) applied during training for LSTM stability
- Chronological train/validation split inside Optuna to prevent lookahead bias
- `SEQUENCE_LENGTH` tunable via environment variable (default 240, shorter sequences like 60 or 120 worth exploring to reduce vanishing gradient risk)

---

### Extension 2: 1D CNN

**Input:** Raw daily return sequence of length `SEQUENCE_LENGTH` (default 240), shape `(n_samples, SEQUENCE_LENGTH)`. Reshaped internally to `(n_samples, 1, SEQUENCE_LENGTH)` for PyTorch Conv1d.

**Architecture:**
```
Input (batch, 1, seq_len)
        ↓
num_layers × [Conv1d + ReLU + MaxPool1d(2)]
        ↓
Global Average Pooling  (batch, num_filters)
        ↓
Dropout
        ↓
Linear (num_filters → 1)
        ↓
Sigmoid → P(outperform cross-sectional median)
```

**Loss function:** Binary cross-entropy

**Hypothesis:**
> The paper's variable importance results show that R(1)–R(5) dominates predictive power. If local short-window patterns are the primary signal, a model whose inductive bias explicitly detects such patterns via learned convolutional filters should be well-matched to this problem and outperform a flat-feature DNN.

**What this tests:** Whether **local short-range pattern detection** is sufficient and better-matched to the dominant signal than either a flat feature vector (DNN) or full sequential memory (LSTM).

**Key implementation details:**
- 1D conv kernels act as learned adaptive sliding windows — analogous to moving averages but with weights optimised for predictive power
- `kernel_size` is the most important hyperparameter — small values (3, 5) focus on short-term patterns matching R(1)–R(5), larger values (10, 20) capture medium-term patterns
- Global Average Pooling chosen over Max Pooling — reflects persistent signal across the sequence rather than detecting a single peak activation
- No gradient clipping needed — CNNs do not suffer from exploding gradients the way LSTMs do
- Note: with 3 conv layers and `MaxPool1d(2)` after each, sequences shorter than ~60 days may compress too heavily before GAP

---

## Hyperparameter Tuning — Optuna

Both the LSTM and CNN support optional Optuna-based hyperparameter tuning controlled by a `use_tuner` flag:

```python
model = LSTMModel(use_tuner=True,  n_trials=50)   # Optuna tunes before training
model = LSTMModel(use_tuner=False)                 # uses fixed default params
```

When `use_tuner=True`, `fit()` automatically calls `tune()` before training — you never need to call it manually. Tuning uses a chronological validation split of the training data (last 20% held out) and trains for `epochs // 4` per trial for speed. Best parameters are saved to `configs/lstm_best_params.json` / `configs/cnn_best_params.json` for reproducibility.

**LSTM search space:**
| Parameter | Values |
|-----------|--------|
| `hidden_size` | 32, 64, 128, 256 |
| `n_layers` | 1 → 3 |
| `dropout` | 0.2 → 0.5 |
| `lr` | 1e-4 → 1e-3 (log scale) |
| `batch_size` | 256, 512, 1024 |

**CNN search space:**
| Parameter | Values |
|-----------|--------|
| `num_filters` | 32, 64, 128 |
| `kernel_size` | 3, 5, 10, 20 |
| `num_layers` | 1 → 3 |
| `dropout` | 0.2 → 0.5 |
| `lr` | 1e-4 → 1e-3 (log scale) |
| `batch_size` | 256, 512, 1024 |

This directly addresses a weakness in the original paper — Krauss et al. acknowledged their DNN used fixed hyperparameters without tuning.

---

## BaseModel Interface

Every model — DNN, GBT, RAF, LSTM, CNN — inherits from `BaseModel` (`src/models/base.py`) and implements the same interface:

```python
model.fit(X_train, y_train)      # train on one sliding window
model.predict_proba(X_trade)     # returns np.ndarray of shape (n_samples,)
model.tune(X_train, y_train)     # optional — only LSTM and CNN implement this
```

The backtest engine interacts exclusively through this interface and has no knowledge of model internals. Both the LSTM and CNN output `P(outperform) ∈ (0, 1)` via a sigmoid final layer, slotting directly into the trading logic:

1. On each trading day, run inference for all ~500 stocks
2. Rank stocks by `P(outperform)` in descending order
3. Go **long top-k**, **short bottom-k**
4. Evaluate using the same metrics as the paper

Transaction costs of 0.05% per share per half-turn applied identically to all models.

---

## Model Comparison Summary

| Model | Input | Architecture | Core Hypothesis |
|-------|-------|--------------|-----------------|
| DNN (baseline) | 31 lagged features | Feedforward, 31-31-10-5-2 | Hand-crafted features + depth sufficient |
| GBT (baseline) | 31 lagged features | Boosted shallow trees | Weak learner ensemble on feature space |
| RAF (baseline) | 31 lagged features | 1,000 deep decorrelated trees | Robust to noisy features via randomization |
| **LSTM (ours)** | Raw sequence (default 240 days) | Stacked LSTM + sigmoid | Learned temporal features > hand-crafted lags |
| **CNN (ours)** | Raw sequence (default 240 days) | Stacked 1D Conv + GAP + sigmoid | Local short-range patterns dominate signal |

---

## Repository Structure

```
├── src/
│   ├── __init__.py
│   ├── processing/                     # Data preparation layer
│   │   ├── __init__.py
│   │   ├── download_wrds.py            # CRSP download via WRDS
│   │   ├── data_processing.py          # Cleaning, constituent matrix, valid universe
│   │   ├── feature_engineering.py      # 31 lagged features for baseline models
│   │   ├── sequence_engineering.py     # Raw sequences for LSTM & CNN
│   │   └── label_engineering.py        # Cross-sectional median binary label (shared)
│   │
│   ├── models/                         # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py                     # Abstract base class — fit() / predict_proba() / tune()
│   │   ├── dnn.py                      # Baseline DNN replication 
│   │   ├── gbt.py                      # Gradient-boosted trees 
│   │   ├── random_forest.py            # Random forest 
│   │   ├── lstm.py                     # LSTM extension 
│   │   └── cnn.py                      # 1D CNN extension 
│   │
│   ├── backtest/                       # Evaluation layer 
│   │   ├── __init__.py
│   │
│   └── app/                            # Streamlit dashboard
│
├── data/
│   ├── raw/                            # Raw CRSP downloads (parquet)
│   │   ├── sp500_constituents.parquet
│   │   └── daily_returns.parquet
│   └── processed/                      # Cleaned data and model inputs (parquet)
│       ├── returns_clean.parquet
│       ├── constituent_matrix.parquet
│       ├── valid_universe.parquet
│       └── features/                   # Per-batch feature and sequence files
│           ├── batch_00_X_train.parquet
│           ├── batch_00_y_train.parquet
│           ├── batch_00_X_trade.parquet
│           ├── batch_00_y_trade.parquet
│           ├── batch_00_meta_trade.parquet
│           ├── ...                     # (23 batches × 5 files = 115 files for features)
│           ├── seq_batch_00_X_train.parquet
│           └── ...                     # (23 batches × 5 files = 115 files for sequences)
│
├── results/
│ 
├── configs/
│   ├── config.yaml                     # All hyperparams, window sizes, cost assumptions
│   ├── lstm_best_params.json           # Best params from Optuna (generated at runtime)
│   └── cnn_best_params.json            # Best params from Optuna (generated at runtime)
│
├── main.py                              # Single entry point — full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
pip install -r requirements.txt
```

Set WRDS credentials before running the data download:
```bash
export WRDS_USERNAME=your_username
```

Run the full pipeline:
```bash
python main.py
```

---

## Requirements

- Python 3.12+
- PyTorch
- Optuna
- pandas, numpy
- wrds
- scikit-learn
- streamlit
- plotly

---

## References

Krauss, C., Do, X. A., & Huck, N. (2016). *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500.* FAU Discussion Papers in Economics, No. 03/2016.