"""
Feature engineering for statistical arbitrage on the S&P 500.
Reproduces Section 4.1 and 4.2 of Krauss, Do & Huck (2016).

Inputs (data/processed/):
    returns_clean.parquet — full returns matrix (date × permno)
    valid_universe.parquet — binary mask: 1 if stock usable on that date

Outputs (data/features/):
    batch_{i:02d}_X_train.parquet — feature matrix for training window i
    batch_{i:02d}_y_train.parquet — labels for training window i
    batch_{i:02d}_X_trade.parquet — feature matrix for trading window i
    batch_{i:02d}_y_trade.parquet — labels for trading window i
    batch_{i:02d}_meta_trade.parquet — (date, permno) index for trading set

Paper parameters exactly reproduced:
    - 31 features: R(1)–R(20) daily lags + R(40,60,...,240) monthly lags
    - Binary label: 1 if stock outperforms cross-sectional median next day
    - Training window: 750 trading days (~3 years)
    - Trading window: 250 trading days (~1 year)
    - Sliding window advances by 250 days → 23 non-overlapping batches
    - Training start: December 1992  |  end: October 2015
    - No lookahead bias: label for day t uses return at t+1 only
    - No forward-filling; stocks with any NaN in their 31-feature window
      are excluded from that day's cross-section
"""
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

# Directory structure
root = Path(__file__).resolve().parents[1]
proc_dir = root/"data"/"processed"
feat_dir = root/"data"/ "features"
feat_dir.mkdir(parents=True, exist_ok=True)

# Section 4.2: lags m ∈ {1,...,20} ∪ {40,60,...,240} → 31 features total
DAILY_LAGS = list(range(1, 21))                    # 20 lags
MONTHLY_LAGS = list(range(40, 241, 20))            # 11 lags (40,60,...,240)
ALL_LAGS = DAILY_LAGS + MONTHLY_LAGS               # 31 features

TRAIN_WINDOW = 750   # trading days, Section 4.1
TRADE_WINDOW = 250   # trading days, Section 4.1
MAX_LAG = 240   # longest lookback needed to compute any feature

# First date for which we can have a full 750-day training set + 240-day
# feature lookback.  Paper's first trading day is December 1992.
FIRST_TRADING_DAY = "1992-12-01"
LAST_TRADING_DAY = "2015-10-31"

# Helper functions
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned returns and valid-universe mask."""
    print("Loading processed data …")
    returns = pd.read_parquet(proc_dir / "returns_clean.parquet")
    valid = pd.read_parquet(proc_dir / "valid_universe.parquet")

    # Align columns (only permnos present in both)
    common = returns.columns.intersection(valid.columns)
    returns= returns[common]
    valid = valid[common]

    print(f"  Returns : {returns.shape[0]:,} days × {returns.shape[1]:,} permnos")
    print(f"  Universe: {valid.shape[0]:,} days × {valid.shape[1]:,} permnos")
    return returns, valid

def compute_price_index(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Convert returns to a cumulative price index starting at 1.0 on the
    first date.  This lets us compute multi-period returns as simple
    ratios: R(t, m) = P(t) / P(t-m) - 1  (equation 1 in the paper).

    NaN returns propagate forward (cumulative product of NaN = NaN),
    so missing data is preserved correctly.
    """
    print("Building cumulative price index …")
    # (1 + r_t) cumulative product; NaN stays NaN via skipna=False default
    price_index = (1 + returns).cumprod()
    return price_index

def build_features_for_date(t_idx: int,
                            price_index: pd.DataFrame,
                            valid_permnos: pd.Index,) -> pd.DataFrame | None:
    """
    Build the 31-feature row for every valid stock on trading day t.

    Parameters
    ----------
    t_idx        : integer position of date t in price_index.index
    price_index  : cumulative price index (date × permno)
    valid_permnos: which permnos are valid (constituent + history) on date t

    Returns
    -------
    DataFrame  shape (n_valid, 31)  with columns named by lag
    or None if no valid stocks remain after NaN filtering.

    Paper eq. (1): R^s_{t,m} = P^s_t / P^s_{t-m}  - 1
    We require t_idx >= MAX_LAG to guarantee P_{t-240} exists.
    """
    if t_idx < MAX_LAG:
        return None

    P_t = price_index.iloc[t_idx][valid_permnos]       # shape (n_valid,)

    rows = {}
    for m in ALL_LAGS:
        P_tm = price_index.iloc[t_idx - m][valid_permnos]
        rows[f"R_{m}"] = (P_t / P_tm - 1).values

    feat = pd.DataFrame(rows, index=valid_permnos)

    # Drop any stock that has a NaN in ANY of its 31 features.
    # (NaN propagates from missing returns in the price index.)
    feat = feat.dropna()
    return feat if len(feat) > 0 else None


def build_label_for_date(t_idx: int,
                         returns: pd.DataFrame,
                         valid_permnos: pd.Index,) -> pd.Series | None:
    """
    Binary label Y^s_{t+1} = 1 if next-day return of stock s exceeds
    the cross-sectional median return across all valid stocks on day t+1.

    Section 4.2: classification target, no regression.
    No lookahead: we use returns at t+1 only, never beyond.
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
                price_index: pd.DataFrame,
                valid: pd.DataFrame,) -> None:
    """
    Build and save one training-trading batch.

    Each batch produces:
      X_train  (n_train_obs × 31)  — features for all valid stocks in train window
      y_train  (n_train_obs,)      — labels
      X_trade  (n_trade_obs × 31)  — features for trading window
      y_trade  (n_trade_obs,)      — labels
      meta_trade (n_trade_obs × 2) — [date, permno] index for live trading use
    """
    all_dates = returns.index

    def collect_window(date_positions):
        X_parts, y_parts, meta_parts = [], [], []

        for t_idx in tqdm(date_positions, leave=False, desc=f"  batch {batch_idx:02d}"):
            date = all_dates[t_idx]

            # Which permnos are valid on this date?
            valid_on_date = valid.loc[date]
            valid_permnos = valid_on_date[valid_on_date == 1].index

            if len(valid_permnos) == 0:
                continue

            feat  = build_features_for_date(t_idx, price_index, valid_permnos)
            if feat is None:
                continue

            label = build_label_for_date(t_idx, returns, feat.index)
            if label is None:
                continue

            # Align: only keep stocks that have both features and a label
            common = feat.index.intersection(label.index)
            if len(common) == 0:
                continue

            feat  = feat.loc[common]
            label = label.loc[common]

            X_parts.append(feat)
            y_parts.append(label)

            meta = pd.DataFrame({
                "date":   date,
                "permno": common,
            })
            meta_parts.append(meta)

        if not X_parts:
            return None, None, None

        X    = pd.concat(X_parts)
        y    = pd.concat(y_parts)
        meta = pd.concat(meta_parts, ignore_index=True)
        return X, y, meta

    print(f"\nBatch {batch_idx:02d} — building training set ({len(train_date_positions)} days) …")
    X_train, y_train, _ = collect_window(train_date_positions)

    print(f"Batch {batch_idx:02d} — building trading set  ({len(trade_date_positions)} days) …")
    X_trade, y_trade, meta_trade = collect_window(trade_date_positions)

    if X_train is None or X_trade is None:
        print(f"  Skipping batch {batch_idx:02d} — insufficient data.")
        return

    # Save
    prefix = feat_dir / f"batch_{batch_idx:02d}"
    X_train.to_parquet(f"{prefix}_X_train.parquet")
    y_train.to_parquet(f"{prefix}_y_train.parquet")
    X_trade.to_parquet(f"{prefix}_X_trade.parquet")
    y_trade.to_parquet(f"{prefix}_y_trade.parquet")
    meta_trade.to_parquet(f"{prefix}_meta_trade.parquet")

    print(f"  Train : {X_train.shape[0]:>8,} obs  |  "
          f"label balance {y_train.mean():.3f}")
    print(f"  Trade : {X_trade.shape[0]:>8,} obs  |  "
          f"label balance {y_trade.mean():.3f}")
    print(f"  Saved → {prefix}_*.parquet")


def main():
    returns, valid = load_data()
    price_index = compute_price_index(returns)

    # We need t_idx >= MAX_LAG (240) to compute all features.
    # The paper's first trading day is Dec 1992.
    all_dates = returns.index

    first_trade_idx = all_dates.searchsorted(FIRST_TRADING_DAY)
    last_trade_idx = all_dates.searchsorted(LAST_TRADING_DAY, side="right") - 1

    # Guard: ensure we can look back MAX_LAG days from the first trading day
    first_trade_idx = max(first_trade_idx, MAX_LAG)

    trading_period_positions = list(range(first_trade_idx, last_trade_idx + 1))
    total_trading_days = len(trading_period_positions)
    print(f"\nTrading period: {all_dates[first_trade_idx].date()} → "
          f"{all_dates[last_trade_idx].date()} ({total_trading_days} days)")

    # Section 4.1: train=750, trade=250, slide by 250 → 23 batches
    # Each batch uses the 750 days immediately before its trading window
    # as its training set.  The training window slides with the batch.
    batches = []
    pos = 0
    while pos + TRADE_WINDOW <= total_trading_days:
        trade_pos = trading_period_positions[pos: pos + TRADE_WINDOW]

        # Training window: 750 days ending the day before the trading window
        trade_start_global = trade_pos[0]          # position in all_dates
        train_end_global = trade_start_global - 1
        train_start_global = train_end_global - TRAIN_WINDOW + 1
        train_start_global = max(train_start_global, MAX_LAG)  # guard against feature lookback

        if train_start_global < 0:
            pos += TRADE_WINDOW
            continue

        train_pos = list(range(train_start_global, train_end_global + 1))
        batches.append((train_pos, trade_pos))
        pos += TRADE_WINDOW

    print(f"Total batches: {len(batches)}  (paper: 23)\n")

    for i, (train_pos, trade_pos) in enumerate(batches):
        train_start = all_dates[train_pos[0]].date()
        train_end = all_dates[train_pos[-1]].date()
        trade_start = all_dates[trade_pos[0]].date()
        trade_end = all_dates[trade_pos[-1]].date()

        print(f"\n{'─'*60}")
        print(f"Batch {i:02d}  |  train {train_start} → {train_end}"
              f"  |  trade {trade_start} → {trade_end}")

        build_batch(batch_idx=i,
                    train_date_positions=train_pos,
                    trade_date_positions=trade_pos,
                    returns=returns,
                    price_index=price_index,
                    valid=valid)

    print("\n\nFeature engineering complete.")
    print(f"Output files saved to: {feat_dir}")


if __name__ == "__main__":
    main()