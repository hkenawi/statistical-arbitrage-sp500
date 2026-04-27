"""
Validate feature and sequence batch parquet files.

Checks:
    - Shape (n_obs, 31) for features, (n_obs, 240) for sequences
    - Column names R_1...R_240 for features, t_0...t_239 for sequences
    - Index is permno
    - Zero NaNs after dropna logic
    - Label balance close to 0.5
    - Value ranges are small decimals
    - n_obs sanity (~375,000 train, ~125,000 trade per batch)
"""
import pandas as pd
from pathlib import Path

feat_dir = Path(__file__).resolve().parents[2]/"data"/"processed"/"features"

# Pick batch 01 — batch 00 skips so 01 is the first real batch
BATCH = "01"

def check_X(path: Path, expected_cols: int, col_prefix: str, label: str):
    print(f"{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")

    if not path.exists():
        print(f"  FILE NOT FOUND: {path}")
        return

    df = pd.read_parquet(path)
    print(f"Shape          : {df.shape}  (n_obs × n_features)")
    print(f"Expected cols  : {expected_cols}")
    print(f"Cols match     : {df.shape[1] == expected_cols}")
    print(f"Index name     : {df.index.name}  (expect: permno)")
    print(f"Index dtype    : {df.index.dtype}  (expect: int64)")
    print(f"NaN count      : {df.isna().sum().sum()}  (must be 0)")

    # Column name check
    expected_first = f"{col_prefix}_1" if col_prefix == "R" else f"{col_prefix}_0"
    expected_last  = f"{col_prefix}_240" if col_prefix == "R" else f"{col_prefix}_239"
    print(f"First col      : {df.columns[0]}  (expect: {expected_first})")
    print(f"Last col       : {df.columns[-1]}  (expect: {expected_last})")

    # Value range
    print(f"Value min      : {df.min().min():.6f}")
    print(f"Value max      : {df.max().max():.6f}")
    print(f"Value mean     : {df.mean().mean():.6f}")
    extreme = (df.abs() > 1).sum().sum()
    print(f"Values > 100%  : {extreme}  (should be near 0)")

    # n_obs sanity
    # paper: ~500 stocks × 750 train days = ~375,000 train obs
    #        ~500 stocks × 250 trade days = ~125,000 trade obs
    n = df.shape[0]
    if "train" in label.lower():
        print(f"n_obs          : {n:,}  (paper expects ~375,000)")
    else:
        print(f"n_obs          : {n:,}  (paper expects ~125,000)")

    # Unique permnos
    if df.index.name == "permno" or "permno" in str(df.index.name):
        n_permnos = df.index.nunique()
        print(f"Unique permnos : {n_permnos}  (expect ~490-500)")
    print()


def check_y(path: Path, label: str):
    print(f"{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")

    if not path.exists():
        print(f"  FILE NOT FOUND: {path}")
        return

    df = pd.read_parquet(path)
    s = df.squeeze()  # Series
    print(f"Shape          : {df.shape}")
    print(f"Column name    : {df.columns.tolist()}  (expect: ['label'])")
    print(f"Unique values  : {sorted(s.unique().tolist())}  (must be [0, 1])")
    balance = s.mean()
    print(f"Label balance  : {balance:.4f}  (expect 0.45–0.55)")
    if not (0.45 <= balance <= 0.55):
        print(f"  WARNING: label balance {balance:.4f} is far from 0.5!")
    print(f"NaN count      : {s.isna().sum()}  (must be 0)")
    print(f"n_obs          : {len(s):,}")
    print()


# ── Feature batches (31-lag baselines) ───────────────────────────────────────
check_X(
    path=feat_dir / f"batch_{BATCH}_X_train.parquet",
    expected_cols=31,
    col_prefix="R",
    label=f"batch_{BATCH}_X_train  (31-lag features, training set)",
)
check_y(
    path=feat_dir / f"batch_{BATCH}_y_train.parquet",
    label=f"batch_{BATCH}_y_train  (labels, training set)",
)
check_X(
    path=feat_dir / f"batch_{BATCH}_X_trade.parquet",
    expected_cols=31,
    col_prefix="R",
    label=f"batch_{BATCH}_X_trade  (31-lag features, trading set)",
)
check_y(
    path=feat_dir / f"batch_{BATCH}_y_trade.parquet",
    label=f"batch_{BATCH}_y_trade  (labels, trading set)",
)

# ── Sequence batches (LSTM / CNN) ─────────────────────────────────────────────
check_X(
    path=feat_dir / f"seq_batch_{BATCH}_X_train.parquet",
    expected_cols=240,
    col_prefix="t",
    label=f"seq_batch_{BATCH}_X_train  (raw sequences, training set)",
)
check_y(
    path=feat_dir / f"seq_batch_{BATCH}_y_train.parquet",
    label=f"seq_batch_{BATCH}_y_train  (labels, training set)",
)
check_X(
    path=feat_dir / f"seq_batch_{BATCH}_X_trade.parquet",
    expected_cols=240,
    col_prefix="t",
    label=f"seq_batch_{BATCH}_X_trade  (raw sequences, trading set)",
)
check_y(
    path=feat_dir / f"seq_batch_{BATCH}_y_trade.parquet",
    label=f"seq_batch_{BATCH}_y_trade  (labels, trading set)",
)

# ── Meta trade files ──────────────────────────────────────────────────────────
def check_meta(path: Path, label: str):
    print(f"{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")

    if not path.exists():
        print(f"  FILE NOT FOUND: {path}")
        return

    df = pd.read_parquet(path)
    print(f"Shape          : {df.shape}")
    print(f"Columns        : {df.columns.tolist()}  (expect: ['date', 'permno'])")
    print(f"Date dtype     : {df['date'].dtype}")
    print(f"Permno dtype   : {df['permno'].dtype}  (expect: int64)")
    print(f"Date range     : {df['date'].min()} → {df['date'].max()}")
    print(f"Unique dates   : {df['date'].nunique()}  (expect ~250 trading days)")
    print(f"Unique permnos : {df['permno'].nunique()}  (expect ~490-500)")
    print(f"NaN count      : {df.isna().sum().sum()}  (must be 0)")
    print(f"n_obs          : {len(df):,}  (expect ~125,000)")

    # Each date should have roughly the same number of stocks
    stocks_per_day = df.groupby("date")["permno"].count()
    print(f"Stocks/day min : {stocks_per_day.min()}  (expect ~490-500)")
    print(f"Stocks/day max : {stocks_per_day.max()}  (expect ~490-500)")
    print(f"Stocks/day mean: {stocks_per_day.mean():.0f}")
    print()


check_meta(
    path=feat_dir / f"batch_{BATCH}_meta_trade.parquet",
    label=f"batch_{BATCH}_meta_trade  (31-lag features)",
)
check_meta(
    path=feat_dir / f"seq_batch_{BATCH}_meta_trade.parquet",
    label=f"seq_batch_{BATCH}_meta_trade  (raw sequences)",
)

print("Done.")