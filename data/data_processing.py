"""
Takes the raw CRSP downloads and produces the clean dataset used for
feature engineering and model training.

Inputs (data/raw/):
    sp500_constituents.parquet
    daily_returns.parquet

Outputs (data/processed/):
    returns_clean.parquet     — returns matrix, only valid trading days,
                                NaN where a stock wasn't yet a constituent
                                or had missing data (NOT forward-filled)
    constituent_matrix.parquet — binary (date × permno): 1 if stock was
                                 an S&P 500 constituent on that date
    valid_universe.parquet    — for each date, which permnos are usable
                                (constituent + has 750 days of prior data)

Key design decisions that match the paper:
  - Survivorship bias: we keep ALL permnos ever in the index, not just
    current members.
  - No forward-filling of returns.  A NaN return means the stock was
    delisted or had a data issue — we mask it out, not fill it.
  - The constituent matrix uses the CRSP membership spells to mark
    exactly which stocks were in the index each day.
  - We do NOT drop stocks with any missing data globally — we only
    require 750 consecutive prior days of data at training time
    (handled in feature engineering, not here).
"""
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

# Setting the directory structure for saving outputs
root = Path(__file__).resolve().parents[1]
raw_dir = root/"data"/"raw"
proc_dir = root/"data"/"processed"
proc_dir.mkdir(parents=True,
               exist_ok=True)

# Paper's trading period starts Dec 1992 (needs 750 days of history from 1990)
TRADING_START = "1992-01-01"
TRADING_END = "2015-10-31"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CRSP data from parquet files"""
    print("Loading raw data...")
    constituents = pd.read_parquet(raw_dir/"sp500_constituents.parquet")
    returns = pd.read_parquet(raw_dir/"daily_returns.parquet")
    print(f"  Returns matrix: {returns.shape[0]:,} days × {returns.shape[1]:,} permnos")
    print(f"  Constituent spells: {len(constituents):,} rows")
    return constituents, returns


def build_constituent_matrix(constituents: pd.DataFrame,
                             date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build a binary (date × permno) DataFrame.
    Entry is 1 if the stock was an S&P 500 constituent on that date.

    Uses the membership spell table from CRSP (start, ending columns).
    When `ending` is NaT, the stock is still in the index.
    """
    print("Building constituent matrix...")
    all_permnos = constituents["permno"].unique()
    matrix = pd.DataFrame(0, index=date_index, columns=all_permnos, dtype=np.int8)

    for _, row in tqdm(constituents.iterrows(), total=len(constituents),
                       desc="  Processing membership spells"):
        permno = row["permno"]
        start  = row["start"]
        end    = row["ending"] if pd.notna(row["ending"]) else date_index[-1]

        mask = (date_index >= start) & (date_index <= end)
        matrix.loc[mask, permno] = 1

    out_path = proc_dir/"constituent_matrix.parquet"
    matrix.to_parquet(out_path)
    print(f"  Saved constituent matrix {matrix.shape} → {out_path}")
    return matrix


def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of the raw returns matrix:
      - Drop columns (permnos) that are entirely NaN
      - Winsorize extreme returns at 1st/99th percentile (per day)
        to handle delisting returns and data errors
      - Keep all dates (full history needed for 750-day lookback)
    """
    print("Cleaning returns...")

    # Drop entirely empty columns
    n_before = returns.shape[1]
    returns = returns.dropna(axis=1, how="all")
    print(f"  Dropped {n_before - returns.shape[1]} all-NaN permnos")

    # Winsorize per day to handle extreme values
    # (e.g. CRSP delisting returns can be -1.0 or very large)
    print("  Winsorizing at 1st/99th percentile per day...")
    lower = returns.quantile(0.01, axis=1)
    upper = returns.quantile(0.99, axis=1)
    returns = returns.clip(lower=lower, upper=upper, axis=0)

    out_path = proc_dir/"returns_clean.parquet"
    returns.to_parquet(out_path)
    print(f"  Saved cleaned returns {returns.shape} → {out_path}")
    return returns


def build_valid_universe(returns: pd.DataFrame,
                         constituent_matrix: pd.DataFrame,
                         min_history: int = 750) -> pd.DataFrame:
    """
    For each date in the trading period, determine which stocks are valid:
      1. Must be a current S&P 500 constituent on that date
      2. Must have at least `min_history` consecutive prior trading days
         of non-NaN returns (required to compute all 31 features)

    Returns a binary (date × permno) DataFrame over the trading period only.
    """
    print(f"Building valid universe (requires {min_history} days of history)...")

    # Align columns — only permnos present in both
    common_permnos = returns.columns.intersection(constituent_matrix.columns)
    returns_aligned = returns[common_permnos]
    constituent_aligned = constituent_matrix[common_permnos]

    # Count consecutive non-NaN days looking backward
    # has_history[t, s] = True if stock s has >= min_history non-NaN returns
    # ending at day t (inclusive)
    not_nan = returns_aligned.notna().astype(int)
    cum_valid = not_nan.cumsum()
    # rolling count of valid days in preceding `min_history` window
    has_history = (
        cum_valid - cum_valid.shift(min_history).fillna(0)
    ) >= min_history

    # Restrict to trading period
    trading_mask = (
        (returns_aligned.index >= TRADING_START) &
        (returns_aligned.index <= TRADING_END)
    )
    has_history_trading = has_history.loc[trading_mask]
    constituent_trading = constituent_aligned.loc[trading_mask]

    # Valid = constituent AND has enough history
    valid_universe = (
        (constituent_trading == 1) & has_history_trading
    ).astype(np.int8)

    out_path = proc_dir/"valid_universe.parquet"
    valid_universe.to_parquet(out_path)

    # Summary stats
    avg_stocks = valid_universe.sum(axis=1).mean()
    print(f"  Average valid stocks per day: {avg_stocks:.0f}")
    print(f"  Saved valid universe {valid_universe.shape} → {out_path}")
    return valid_universe


def print_sanity_checks(returns: pd.DataFrame,
                        constituent_matrix: pd.DataFrame,
                        valid_universe: pd.DataFrame) -> None:
    print("\n── Sanity checks ──────────────────────────────────────")
    print(f"  Total permnos ever in S&P 500:  {returns.shape[1]:,}")
    print(f"  Date range in returns:          {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"  Trading period days:            {valid_universe.shape[0]:,}")

    # Spot check: around 500 stocks should be valid on any given recent date
    sample_date = valid_universe.index[valid_universe.index >= "2010-01-01"][0]
    n_valid = valid_universe.loc[sample_date].sum()
    print(f"  Valid stocks on {sample_date.date()}:      {n_valid}  (expect ~450-500)")

    # Check no lookahead in constituent matrix (start dates)
    print(f"  Constituent matrix covers:      "
          f"{constituent_matrix.index[0].date()} → {constituent_matrix.index[-1].date()}")
    print("────────────────────────────────────────────────────────")


def main():
    constituents, returns = load_raw()

    returns_clean = clean_returns(returns)

    # Use the full date index from returns for constituent matrix
    constituent_matrix = build_constituent_matrix(constituents, returns_clean.index)

    valid_universe = build_valid_universe(returns_clean, constituent_matrix)

    print_sanity_checks(returns_clean, constituent_matrix, valid_universe)
    print("\nDataset build complete.")


if __name__ == "__main__":
    main()