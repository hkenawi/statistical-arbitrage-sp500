"""
Trains one RandomForestModel per sliding-window batch (23 total).
All training logic is encapsulated in RandomForestModel.fit() — this script
is responsible only for data loading, model instantiation, and
orchestrating the batch loop.

Outputs per batch:
    src/train/checkpoints/random_forest/batch_{i:02d}/latest.joblib        — saved every checkpoint interval
    src/train/checkpoints/random_forest/batch_{i:02d}/trees_NNNN.joblib    — every N trees
    src/train/checkpoints/random_forest/batch_{i:02d}/final.joblib         — end of training
    src/train/checkpoints/random_forest/batch_{i:02d}/metadata.json        — hyperparams + timestamp

Note on checkpoint_every for RAF:
    The paper's default is 1000 trees. With checkpoint_every=5 this produces
    200 snapshots per batch — each a full serialized forest that grows on disk.
    Consider setting checkpoint_every=50 or 100 in config.yaml for the full run
    to keep disk usage reasonable (each snapshot is ~100–300 MB).

Standalone usage:
    python -m src.train.train_random_forest
"""

import sys
import yaml
import numpy as np

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.models.random_forest import RandomForestModel

_FEAT_DIR = _ROOT/"data"/"processed"/"features"
_CKPT_DIR = Path(__file__).resolve().parent/"checkpoints"/"random_forest"


def load_config() -> dict:
    with open(_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def train_batch(batch_idx: int, cfg: dict) -> None:
    """Train Random Forest on one sliding-window batch."""
    import pandas as pd

    x_path = _FEAT_DIR / f"batch_{batch_idx:02d}_X_train.parquet"
    y_path = _FEAT_DIR / f"batch_{batch_idx:02d}_y_train.parquet"

    if not x_path.exists():
        print(f"  Batch {batch_idx:02d}: data not found, skipping.")
        return

    X_train = pd.read_parquet(x_path)
    y_train = pd.read_parquet(y_path).squeeze()

    raf_cfg = cfg["models"]["random_forest"]

    model = RandomForestModel(n_trees=raf_cfg.get("n_trees", 1000),
                              max_depth=raf_cfg.get("max_depth", 20),
                              col_sample=raf_cfg.get("col_sample", "sqrt"),
                              seed=raf_cfg.get("seed", 1),
                              use_tuner=raf_cfg.get("use_tuner", False),
                              n_trials=raf_cfg.get("n_trials", 50),
                              val_fraction=raf_cfg.get("val_fraction", 0.2),
                              n_jobs=raf_cfg.get("n_jobs", -1),)

    ckpt_dir = _CKPT_DIR/f"batch_{batch_idx:02d}"

    model.fit(X_train,
              y_train,
              checkpoint_dir=ckpt_dir,
              checkpoint_every=cfg["training"]["checkpoint_every"],)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["reproducibility"]["global_seed"])

    n_batches = cfg["windows"]["n_batches"]

    print(f"Random Forest training — {n_batches} batches")
    print(f"Checkpoints → {_CKPT_DIR}\n")

    for batch_idx in range(n_batches):
        print(f"\n{'─' * 60}")
        print(f"Batch {batch_idx:02d}/{n_batches - 1}")
        train_batch(batch_idx, cfg)

    print("\nRandom Forest training complete.")


if __name__ == "__main__":
    main()