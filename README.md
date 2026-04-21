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
The paper's variable importance analysis revealed that **R(1)–R(5) — the past 5 trading days — carry the highest predictive power** across all three base models. Returns in the R(10)–R(20) range rank lowest. This finding directly motivates the architectural choices of our extensions.

---

## Our Extensions

### Why Not Just Use the 31 Features?

The original paper's 31 features are a hand-crafted, multi-resolution encoding of the return history — a deliberate design choice that collapses the full time series into a fixed-size snapshot. While effective, this representation makes two implicit assumptions:

1. The chosen lag intervals (daily for 20 days, then monthly) are the right resolution
2. The temporal ordering and sequential dependencies within the series are not important beyond what these aggregated features capture

Our extensions challenge both assumptions by feeding **raw return sequences directly** into architectures whose inductive biases are designed for sequential data.

---

### Extension 1: LSTM

**Input:** Raw daily return sequence of length 240 (one trading year), shape `(240, 1)` per stock per day

**Architecture:**
```
Raw return sequence (240 × 1)
        ↓
LSTM layer(s)
        ↓
Final hidden state h_T
        ↓
Dense layer (optional)
        ↓
Sigmoid output → P(outperform cross-sectional median)
```

**Loss function:** Binary cross-entropy (identical to original DNN)

**Hypothesis:**
> Hand-crafted lag features, while a reasonable approximation, cannot fully capture the non-linear sequential dependencies present in raw return history. An LSTM, which maintains a hidden state across the full 240-day sequence, can learn which temporal patterns are predictive without requiring pre-specified lag intervals — and should therefore outperform the 31-feature DNN.

**What this tests:** Whether **long-range sequential memory and learned temporal feature extraction** outperform fixed, hand-engineered lag aggregation.

---

### Extension 2: 1D CNN

**Input:** Raw daily return sequence of length 240, shape `(240, 1)` per stock per day

**Architecture:**
```
Raw return sequence (240 × 1)
        ↓
1D Convolutional layers (+ ReLU + pooling)
        ↓
Flatten or Global Average Pooling
        ↓
Dense layer (optional)
        ↓
Sigmoid output → P(outperform cross-sectional median)
```

**Loss function:** Binary cross-entropy

**Hypothesis:**
> The paper's own variable importance results show that short-term returns (R(1)–R(5)) dominate predictive power. If local short-window patterns are the primary signal, then a model whose inductive bias is explicitly designed to detect such patterns — via learned convolutional filters acting as adaptive, overlapping moving averages — should be well-matched to this problem and outperform a flat-feature DNN.

**What this tests:** Whether **local short-range pattern detection** is sufficient and better-matched to the dominant signal in the data than either a flat feature vector (DNN) or full sequential memory (LSTM).

**Note on convolution intuition:** A 1D convolutional kernel computes the dot product between a learned transformation matrix and a sliding window of the return sequence. This is analogous to a learned, adaptive moving average — but unlike a fixed moving average, the kernel weights are optimized to detect patterns that are actually predictive of next-day relative performance.

---

## How Both Models Plug Into the Original Framework

Both the LSTM and CNN output a scalar `P(outperform) ∈ (0, 1)` via a sigmoid final layer — directly comparable to the probability forecasts produced by the original DNN, GBT, and RAF. This means both models slot cleanly into Krauss et al.'s trading logic:

1. On each trading day, run inference for all ~500 stocks
2. Rank stocks by `P(outperform)` in descending order
3. Go **long top-k**, **short bottom-k**
4. Evaluate using the same metrics: mean daily return, Sharpe ratio, max drawdown, Calmar ratio, VaR, CVaR, and Fama-French alpha regressions

Transaction costs of 0.05% per share per half-turn are applied identically to all models.

---

## Model Comparison Summary

| Model | Input | Architecture | Core Hypothesis |
|-------|-------|--------------|-----------------|
| DNN (baseline) | 31 lagged features | Feedforward, 31-31-10-5-2 | Hand-crafted features + depth sufficient |
| GBT (baseline) | 31 lagged features | Boosted shallow trees | Weak learner ensemble on feature space |
| RAF (baseline) | 31 lagged features | 1,000 deep decorrelated trees | Robust to noisy features via randomization |
| **LSTM (ours)** | 240-day raw sequence | Recurrent, sequential memory | Learned temporal features > hand-crafted lags |
| **CNN (ours)** | 240-day raw sequence | 1D convolutions, local patterns | Local short-range patterns dominate signal |

---

## Repository Structure

```
├── data/
│   └── sp500_returns.csv          # Daily total return indices (from Datastream)
├── preprocessing/
│   ├── survivor_bias.py           # S&P 500 constituent list handling
│   ├── feature_engineering.py     # 31 lagged features (for baseline replication)
│   └── sequence_builder.py        # Raw 240-day sequences (for LSTM & CNN)
├── models/
│   ├── dnn.py                     # Baseline DNN replication
│   ├── gbt.py                     # Gradient-boosted trees
│   ├── random_forest.py           # Random forest
│   ├── lstm.py                    # LSTM extension
│   └── cnn.py                     # 1D CNN extension
├── backtest/
│   ├── engine.py                  # Long-short portfolio construction
│   ├── transaction_costs.py       # 0.05% per half-turn cost model
│   └── performance.py             # Sharpe, drawdown, VaR, FF-alpha
├── results/
│   └── ...                        # Output tables and figures
└── README.md
```

---

## References

Krauss, C., Do, X. A., & Huck, N. (2016). *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500.* FAU Discussion Papers in Economics, No. 03/2016.