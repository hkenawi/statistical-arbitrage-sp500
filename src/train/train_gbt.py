"""
Trains one GBTModel per sliding-window batch (23 total).
All training logic is encapsulated in GBTModel.fit() — this script
is responsible only for data loading, model instantiation, and
orchestrating the batch loop.

Outputs per batch:
    src/train/checkpoints/gbt/batch_{i:02d}/latest.joblib      — saved every checkpoint interval
    src/train/checkpoints/gbt/batch_{i:02d}/round_NNN.joblib   — every N boosting rounds
    src/train/checkpoints/gbt/batch_{i:02d}/final.joblib       — end of training
    src/train/checkpoints/gbt/batch_{i:02d}/metadata.json      — hyperparams + timestamp

Standalone usage:
    python -m src.train.train_gbt
"""

import sys
import yaml
import numpy as np

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.models.gbt import GBTModel

_FEAT_DIR = _ROOT / "data" / "processed" / "features"
_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "gbt"


def load_config() -> dict:
    with open(_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def train_batch(batch_idx: int, cfg: dict) -> None:
    """Train GBT on one sliding-window batch."""
    import pandas as pd

    x_path = _FEAT_DIR / f"batch_{batch_idx:02d}_X_train.parquet"
    y_path = _FEAT_DIR / f"batch_{batch_idx:02d}_y_train.parquet"

    if not x_path.exists():
        print(f"  Batch {batch_idx:02d}: data not found, skipping.")
        return

    X_train = pd.read_parquet(x_path)
    y_train = pd.read_parquet(y_path).squeeze()

    gbt_cfg = cfg["models"]["gbt"]

    model = GBTModel(
        n_trees=gbt_cfg.get("n_trees", 100),
        max_depth=gbt_cfg.get("max_depth", 3),
        learning_rate=gbt_cfg.get("learning_rate", 0.1),
        col_sample=gbt_cfg.get("col_sample", 0.5),
        seed=gbt_cfg.get("seed", 1),
        use_tuner=gbt_cfg.get("use_tuner", False),
        n_trials=gbt_cfg.get("n_trials", 50),
        val_fraction=gbt_cfg.get("val_fraction", 0.2),
    )

    ckpt_dir = _CKPT_DIR / f"batch_{batch_idx:02d}"

    model.fit(
        X_train,
        y_train,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=cfg["training"]["checkpoint_every"],
    )


def main() -> None:
    cfg = load_config()
    set_seed(cfg["reproducibility"]["global_seed"])

    n_batches = cfg["windows"]["n_batches"]

    print(f"GBT training — {n_batches} batches")
    print(f"Checkpoints → {_CKPT_DIR}\n")

    for batch_idx in range(n_batches):
        print(f"\n{'─' * 60}")
        print(f"Batch {batch_idx:02d} / {n_batches - 1}")
        train_batch(batch_idx, cfg)

    print("\nGBT training complete.")


if __name__ == "__main__":
    main()