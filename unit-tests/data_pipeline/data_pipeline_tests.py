"""
Validate the three processed parquet files.

Checks:
    - Shape and date ranges
    - NaN presence and return value ranges
    - Constituent matrix binary integrity
    - Valid universe subset of constituent matrix (no leakage)
    - Alignment across all three files
    - First date with valid stocks (should be well before Dec 1992)
"""
import pandas as pd
from pathlib import Path

proc_dir = Path(__file__).resolve().parents[2]/"data"/"processed"

# ── returns_clean ─────────────────────────────────────────────────────────────
print("=" * 60)
print("returns_clean.parquet")
print("=" * 60)
ret = pd.read_parquet(proc_dir / "returns_clean.parquet")
print(f"Shape          : {ret.shape}  (days × permnos)")
print(f"Date range     : {ret.index[0].date()} → {ret.index[-1].date()}")
print(f"Column dtype   : {ret.columns.dtype}")
print(f"NaN count      : {ret.isna().sum().sum():,}  ({ret.isna().mean().mean() * 100:.1f}% of cells)")
print(f"Return min     : {ret.min().min():.4f}")
print(f"Return max     : {ret.max().max():.4f}")
print(f"Return mean    : {ret.mean().mean():.6f}")
extreme = (ret.abs() > 1).sum().sum()
print(f"Returns > 100% : {extreme:,}  (should be near 0 for daily returns)")
print(f"Sample permnos : {list(ret.columns[:5])}")
print()

# ── constituent_matrix ────────────────────────────────────────────────────────
print("=" * 60)
print("constituent_matrix.parquet")
print("=" * 60)
cm = pd.read_parquet(proc_dir / "constituent_matrix.parquet")
print(f"Shape          : {cm.shape}  (days × permnos)")
print(f"Date range     : {cm.index[0].date()} → {cm.index[-1].date()}")
print(f"Unique values  : {sorted(cm.stack().unique().tolist())}  (should be [0, 1])")
daily_counts = cm.sum(axis=1)
print(f"Avg stocks/day : {daily_counts.mean():.0f}")
print(f"Min stocks/day : {daily_counts.min():.0f}")
print(f"Max stocks/day : {daily_counts.max():.0f}")
for spot in ["1992-12-01", "2000-01-03", "2010-01-04", "2015-10-01"]:
    if spot in cm.index:
        print(f"Stocks on {spot} : {int(cm.loc[spot].sum())}  (expect ~490-500)")
print()

# ── valid_universe ────────────────────────────────────────────────────────────
print("=" * 60)
print("valid_universe.parquet")
print("=" * 60)
vu = pd.read_parquet(proc_dir / "valid_universe.parquet")
print(f"Shape          : {vu.shape}  (days × permnos)")
print(f"Date range     : {vu.index[0].date()} → {vu.index[-1].date()}")
print(f"Unique values  : {sorted(vu.stack().unique().tolist())}  (should be [0, 1])")
valid_counts = vu.sum(axis=1)
print(f"Avg valid/day  : {valid_counts.mean():.0f}  (paper expects ~500)")
print(f"Min valid/day  : {valid_counts.min():.0f}")
print(f"Max valid/day  : {valid_counts.max():.0f}")

# First date with any valid stocks — critical for batch 00
nonzero = vu.index[valid_counts > 0]
if len(nonzero) > 0:
    first_valid = nonzero[0]
    print(f"First date with valid stocks : {first_valid.date()}  (need before 1992-11-30 for batch 00)")
    print(f"  Valid stocks on that date  : {int(vu.loc[first_valid].sum())}")
else:
    print("WARNING: No valid stocks found on any date!")

# Check specific training end dates for early batches
for d in ["1992-11-30", "1993-11-24", "1994-11-21"]:
    if d in vu.index:
        n = int(vu.loc[d].sum())
        print(f"Valid stocks on {d} (batch train end): {n}  (expect ~490-500, 0 = batch will skip)")
    else:
        print(f"{d} NOT in valid_universe index — batch will skip")
print()

# ── Alignment checks ──────────────────────────────────────────────────────────
print("=" * 60)
print("Alignment checks")
print("=" * 60)
print(f"returns_clean cols == constituent_matrix cols : {ret.columns.equals(cm.columns)}")
print(f"returns_clean cols == valid_universe cols     : {ret.columns.equals(vu.columns)}")
print(f"constituent_matrix covers valid_universe dates: "
      f"{cm.index[0] <= vu.index[0] and cm.index[-1] >= vu.index[-1]}")

# Valid universe must be a subset of constituent matrix —
# no stock should be marked valid without being a constituent
cm_aligned = cm.loc[vu.index]
leaks = ((vu == 1) & (cm_aligned == 0)).sum().sum()
print(f"Valid but not constituent (must be 0): {leaks}")
if leaks > 0:
    print("  WARNING: valid_universe contains stocks not in constituent_matrix!")
    print("  This indicates a bug in build_valid_universe()")

print()
print("Done.")