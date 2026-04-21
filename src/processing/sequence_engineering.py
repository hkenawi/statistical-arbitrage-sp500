"""
Sequence builder for LSTM and CNN models.
Produces raw return sequences instead of the 31 hand-crafted lag features
used in Krauss et al. (2016).

The core hypothesis being tested is that learned temporal feature extraction
over a raw return sequence outperforms fixed, hand-engineered lag aggregation.

Inputs (data/processed/):
    returns_clean.parquet  — full returns matrix (date × permno)
    valid_universe.parquet — binary mask: 1 if stock usable on that date

Outputs (data/processed/features/):
    seq_batch_{i:02d}_X_train.parquet — training sequences, shape (n_obs, SEQUENCE_LENGTH)
    seq_batch_{i:02d}_y_train.parquet — labels for training window i
    seq_batch_{i:02d}_X_trade.parquet — trading sequences, shape (n_obs, SEQUENCE_LENGTH)
    seq_batch_{i:02d}_y_trade.parquet — labels for trading window i
    seq_batch_{i:02d}_meta_trade.parquet — (date, permno) index for trading set

Design decisions:
    - Each row is one stock-day: 240 columns = sequence timesteps t-239 ... t
    - Reshape to (n_obs, SEQUENCE_LENGTH, 1) at model load time
    - Labels are identical to feature_engineering.py — binary cross-sectional
      median — so sequences plug into the same backtest engine as the baselines
    - SEQUENCE_LENGTH is tunable via environment variable (default 240)
    - Same 23-batch sliding window structure as feature_engineering.py
    - No forward-filling; stocks with any NaN in their sequence window
      are excluded from that day's cross-section
"""
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

# Directory structure
root = Path(__file__).resolve().parents[1]
proc_dir = root/"data"/"processed"
feat_dir = root/"data"/"processed"/"features"
feat_dir.mkdir(parents=True, exist_ok=True)

# Sequence length is tunable — override via environment variable or config.yaml.
# 240 matches the full feature lookback in the paper, but shorter sequences
# (e.g. 60, 120) reduce vanishing gradient risk and training time for the LSTM.
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", 240))
MAX_LAG = SEQUENCE_LENGTH

TRAIN_WINDOW = 750   # trading days, matches Section 4.1 of the paper
TRADE_WINDOW = 250   # trading days, matches Section 4.1 of the paper

FIRST_TRADING_DAY = "1992-12-01"
LAST_TRADING_DAY = "2015-10-31"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned returns and valid-universe mask."""
    print("Loading processed data...")
    returns = pd.read_parquet(proc_dir / "returns_clean.parquet")
    valid = pd.read_parquet(proc_dir / "valid_universe.parquet")

    common = returns.columns.intersection(valid.columns)
    returns = returns[common]
    valid = valid[common]

    print(f"  Returns : {returns.shape[0]:,} days × {returns.shape[1]:,} permnos")
    print(f"  Universe: {valid.shape[0]:,} days × {valid.shape[1]:,} permnos")
    print(f"  Sequence length: {SEQUENCE_LENGTH} days")
    return returns, valid


def build_sequence_for_date(t_idx: int,
                            returns: pd.DataFrame,
                            valid_permnos: pd.Index) -> pd.DataFrame | None:
    """
    Build the raw return sequence for every valid stock on trading day t.

    For each stock s, the sequence is:
        [r_{t - SEQUENCE_LENGTH + 1}, ..., r_{t-1}, r_t]

    i.e. the most recent SEQUENCE_LENGTH daily returns ending on day t,
    in chronological order so the LSTM sees the oldest return first.

    Parameters
    ----------
    t_idx         : integer position of date t in returns.index
    returns       : full cleaned returns matrix (date × permno)
    valid_permnos : which permnos are valid (constituent + history) on date t

    Returns
    -------
    DataFrame shape (n_valid, SEQUENCE_LENGTH) — one row per stock,
    columns are t_0 ... t_{SEQUENCE_LENGTH-1} (chronological order),
    or None if no valid stocks remain after NaN filtering.
    """
    if t_idx < MAX_LAG:
        return None

    # Slice the window: rows from t-SEQUENCE_LENGTH+1 to t inclusive
    window = returns.iloc[t_idx - SEQUENCE_LENGTH + 1 : t_idx + 1][valid_permnos]

    # Transpose so rows=permnos, columns=timesteps
    seq = window.T  # shape (n_valid, SEQUENCE_LENGTH)
    seq.columns = [f"t_{i}" for i in range(SEQUENCE_LENGTH)]

    # Drop any stock that has a NaN anywhere in its sequence window
    seq = seq.dropna()
    return seq if len(seq) > 0 else None


def build_label_for_date(t_idx: int,
                         returns: pd.DataFrame,
                         valid_permnos: pd.Index) -> pd.Series | None:
    """
    Binary label Y^s_{t+1} = 1 if next-day return of stock s exceeds
    the cross-sectional median return across all valid stocks on day t+1.

    Identical to feature_engineering.py — ensures sequences plug directly
    into the same backtest engine as the baseline models.

    No lookahead: uses returns at t+1 only, never beyond.
    """
    if t_idx + 1 >= len(returns):
        return None

    r_next = returns.iloc[t_idx + 1][valid_permnos].dropna()
    if len(r_next) == 0:
        return None

    median = r_next.median()
    label = (r_next > median).astype(np.int8)
    label.name = "label"
    return label


def build_batch(batch_idx: int,
                train_date_positions: list[int],
                trade_date_positions: list[int],
                returns: pd.DataFrame,
                valid: pd.DataFrame) -> None:
    """
    Build and save one training-trading batch of sequences.

    Each batch produces:
      X_train     (n_train_obs × SEQUENCE_LENGTH) — raw sequences, train window
      y_train     (n_train_obs,)                  — labels
      X_trade     (n_trade_obs × SEQUENCE_LENGTH) — raw sequences, trade window
      y_trade     (n_trade_obs,)                  — labels
      meta_trade  (n_trade_obs × 2)               — [date, permno] for backtest
    """
    all_dates = returns.index

    def collect_window(date_positions):
        X_parts, y_parts, meta_parts = [], [], []

        for t_idx in tqdm(date_positions, leave=False, desc=f"  batch {batch_idx:02d}"):
            date = all_dates[t_idx]

            valid_on_date = valid.loc[date]
            valid_permnos = valid_on_date[valid_on_date == 1].index

            if len(valid_permnos) == 0:
                continue

            seq = build_sequence_for_date(t_idx, returns, valid_permnos)
            if seq is None:
                continue

            label = build_label_for_date(t_idx, returns, seq.index)
            if label is None:
                continue

            # Align: only keep stocks that have both a sequence and a label
            common = seq.index.intersection(label.index)
            if len(common) == 0:
                continue

            seq = seq.loc[common]
            label = label.loc[common]

            X_parts.append(seq)
            y_parts.append(label)
            meta_parts.append(pd.DataFrame({"date":   date,
                                            "permno": common,
                                            }))

        if not X_parts:
            return None, None, None

        X = pd.concat(X_parts)
        y = pd.concat(y_parts)
        meta = pd.concat(meta_parts, ignore_index=True)
        return X, y, meta

    print(f"\nBatch {batch_idx:02d} — building training set ({len(train_date_positions)} days)...")
    X_train, y_train, _ = collect_window(train_date_positions)

    print(f"Batch {batch_idx:02d} — building trading set  ({len(trade_date_positions)} days)...")
    X_trade, y_trade, meta_trade = collect_window(trade_date_positions)

    if X_train is None or X_trade is None:
        print(f"  Skipping batch {batch_idx:02d} — insufficient data.")
        return

    # Save — prefixed with 'seq_' to distinguish from feature_engineering outputs
    prefix = feat_dir / f"seq_batch_{batch_idx:02d}"
    X_train.to_parquet(f"{prefix}_X_train.parquet")
    y_train.to_parquet(f"{prefix}_y_train.parquet")
    X_trade.to_parquet(f"{prefix}_X_trade.parquet")
    y_trade.to_parquet(f"{prefix}_y_trade.parquet")
    meta_trade.to_parquet(f"{prefix}_meta_trade.parquet")

    balance_train = y_train.mean()
    balance_trade = y_trade.mean()

    if not (0.45 <= balance_train <= 0.55):
        print(f"  WARNING: train label balance {balance_train:.3f} is far from 0.5 — check upstream data")
    else:
        print(f"  Train : {X_train.shape[0]:>8,} obs  |  label balance {balance_train:.3f} ✓")

    if not (0.45 <= balance_trade <= 0.55):
        print(f"  WARNING: trade label balance {balance_trade:.3f} is far from 0.5 — check upstream data")
    else:
        print(f"  Trade : {X_trade.shape[0]:>8,} obs  |  label balance {balance_trade:.3f} ✓")

    print(f"  Saved → {prefix}_*.parquet")


def main():
    returns, valid = load_data()

    all_dates = returns.index

    first_trade_idx = all_dates.searchsorted(FIRST_TRADING_DAY)
    last_trade_idx  = all_dates.searchsorted(LAST_TRADING_DAY, side="right") - 1

    # Guard: ensure every training day can look back SEQUENCE_LENGTH days
    first_trade_idx = max(first_trade_idx, MAX_LAG)

    trading_period_positions = list(range(first_trade_idx, last_trade_idx + 1))
    total_trading_days = len(trading_period_positions)

    print(f"\nTrading period : {all_dates[first_trade_idx].date()} → "
          f"{all_dates[last_trade_idx].date()} ({total_trading_days} days)")

    # Build 23 sliding window batches — identical structure to feature_engineering.py
    batches = []
    pos = 0
    while pos + TRADE_WINDOW <= total_trading_days:
        trade_pos = trading_period_positions[pos : pos + TRADE_WINDOW]

        trade_start_global = trade_pos[0]
        train_end_global = trade_start_global - 1
        train_start_global = train_end_global - TRAIN_WINDOW + 1
        train_start_global = max(train_start_global, MAX_LAG)  # guard against sequence lookback

        if train_start_global < 0:
            pos += TRADE_WINDOW
            continue

        train_pos = list(range(train_start_global, train_end_global + 1))
        batches.append((train_pos, trade_pos))
        pos += TRADE_WINDOW

    print(f"Total batches  : {len(batches)}  (expect 23)\n")

    for i, (train_pos, trade_pos) in enumerate(batches):
        train_start = all_dates[train_pos[0]].date()
        train_end = all_dates[train_pos[-1]].date()
        trade_start = all_dates[trade_pos[0]].date()
        trade_end = all_dates[trade_pos[-1]].date()

        print(f"\n{'─' * 60}")
        print(f"Batch {i:02d}  |  train {train_start} → {train_end}"
              f"  |  trade {trade_start} → {trade_end}")

        build_batch(batch_idx=i,
                    train_date_positions=train_pos,
                    trade_date_positions=trade_pos,
                    returns=returns,
                    valid=valid)

    print("\n\nSequence building complete.")
    print(f"Output files saved to : {feat_dir}")
    print(f"Sequence length used  : {SEQUENCE_LENGTH} days")
    print(f"Reshape to (n, {SEQUENCE_LENGTH}, 1) at model load time.")


if __name__ == "__main__":
    main()