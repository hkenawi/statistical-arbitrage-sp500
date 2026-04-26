"""
Downloads all data needed to reproduce Krauss et al. (2016) using WRDS/CRSP.

Two things are pulled:
  1. S&P 500 historical constituent lists  (CRSP: crsp.msp500list)
  2. Daily total return data for every stock that was ever a constituent
     (CRSP: crsp.dsf joined with crsp.msenames)

Output files (written to data/raw/):
  sp500_constituents.parquet  — one row per (permno, start, end) membership spell
  daily_returns.parquet       — daily total returns, wide format (date x permno)

Requirements:
  WRDS credentials set via environment variables
"""
import os
import wrds
import argparse
import pandas as pd

from tqdm import tqdm
from pathlib import Path

# Setting the directory
root = Path(__file__).resolve().parents[2]
raw_dir = root/"data"/"raw"
raw_dir.mkdir(parents=True,
              exist_ok=True)

# These dates match the paper
DEFAULT_START = "1990-01-01"
DEFAULT_END = "2015-10-31"


def connect_wrds() -> wrds.Connection:
    """Open a WRDS connection"""
    username = os.environ.get("WRDS_USERNAME")
    return wrds.Connection(wrds_username=username)


def download_constituents(db: wrds.Connection,
                          start: str,
                          end: str) -> pd.DataFrame:
    """
    Pull S&P 500 membership spells from crsp.msp500list.

    Returns a DataFrame with columns:
        permno  — CRSP permanent security identifier
        start   — first date the stock was in the index
        ending  — last date (or NaT if still in)
    """
    print("Downloading S&P 500 constituent history from CRSP...")
    query = f"""
        SELECT permno, start, ending
        FROM crsp.msp500list
        WHERE start <= '{end}'
          AND (ending >= '{start}' OR ending IS NULL)
        ORDER BY permno, start
    """
    df = db.raw_sql(query, date_cols=["start", "ending"])
    out_path = raw_dir/"sp500_constituents.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df):,} membership spells → {out_path}")
    return df


def download_returns(db: wrds.Connection,
                     constituents: pd.DataFrame,
                     start: str,
                     end: str,
                     chunk_size: int = 100) -> pd.DataFrame:
    """
    Pull daily total returns (ret) from crsp.dsf for every permno that was
    ever an S&P 500 constituent.

    Uses `ret` (holding-period return including dividends) which is the CRSP
    equivalent of Datastream's total return index — it accounts for dividends,
    splits, and all corporate actions.

    Returns a wide DataFrame: index=date, columns=permno (int).
    """
    permnos = constituents["permno"].unique().tolist()
    print(f"  {len(permnos):,} unique permnos to download")

    chunks = [permnos[i:i + chunk_size] for i in range(0, len(permnos), chunk_size)]
    frames = []

    for chunk in tqdm(chunks, desc="Downloading return chunks"):
        perm_str = ", ".join(str(p) for p in chunk)
        query = f"""
            SELECT date, permno, ret
            FROM crsp.dsf
            WHERE permno IN ({perm_str})
              AND date >= '{start}'
              AND date <= '{end}'
        """
        df_chunk = db.raw_sql(query, date_cols=["date"])
        frames.append(df_chunk)

    print("Concatenating chunks...")
    long = pd.concat(frames, ignore_index=True)

    # pivot to wide: rows=date, cols=permno
    print("Pivoting to wide format (date × permno)...")
    wide = long.pivot(index="date", columns="permno", values="ret")
    wide.sort_index(inplace=True)
    wide.columns = wide.columns.astype(int)

    out_path = raw_dir/"daily_returns.parquet"
    wide.to_parquet(out_path)
    print(f"  Saved returns matrix {wide.shape} → {out_path}")
    return wide


def main():
    parser = argparse.ArgumentParser(description="Download CRSP data for stat-arb project")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=DEFAULT_END,   help="End date YYYY-MM-DD")
    parser.add_argument("--skip-returns",
                        action="store_true",
                        help="Only download constituents (faster for testing)")
    args = parser.parse_args()

    print(f"Date range: {args.start} → {args.end}")
    print(f"Output directory: {raw_dir}\n")

    db = connect_wrds()

    constituents = download_constituents(db, args.start, args.end)

    if not args.skip_returns:
        download_returns(db, constituents, args.start, args.end)

    db.close()
    print("\nDone.")

if __name__ == "__main__":
    main()