"""
Trains one DNNModel per sliding-window batch (23 total).
All training logic is encapsulated in DNNModel.fit() — this script
is responsible only for data loading, model instantiation, and
orchestrating the batch loop.

Outputs per batch:
    src/train/checkpoints/dnn/batch_{i:02d}/latest.pt     — saved every epoch
    src/train/checkpoints/dnn/batch_{i:02d}/epoch_NNN.pt  — every N epochs
    src/train/checkpoints/dnn/batch_{i:02d}/final.pt      — end of training

Standalone usage:
    python -m src.train.train_dnn
"""

import sys
import yaml
import numpy as np
import torch

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.models.dnn import DNNModel

_FEAT_DIR = _ROOT/"data"/"processed"/"features"
_CKPT_DIR = Path(__file__).resolve().parent/"checkpoints"/"dnn"


def load_config() -> dict:
    with open(_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_batch(batch_idx: int, cfg: dict) -> None:
    """Train DNN on one sliding-window batch."""
    import pandas as pd

    x_path = _FEAT_DIR / f"batch_{batch_idx:02d}_X_train.parquet"
    y_path = _FEAT_DIR / f"batch_{batch_idx:02d}_y_train.parquet"

    if not x_path.exists():
        print(f"  Batch {batch_idx:02d}: data not found, skipping.")
        return

    X_train = pd.read_parquet(x_path)
    y_train = pd.read_parquet(y_path).squeeze()

    dnn_cfg = cfg["models"]["dnn"]

    model = DNNModel(
        architecture=dnn_cfg.get("architecture", [31, 31, 10, 5, 2]),
        dropout_hidden=dnn_cfg.get("dropout_hidden", 0.5),
        dropout_input=dnn_cfg.get("dropout_input", 0.1),
        l1_lambda=dnn_cfg.get("l1_lambda", 1e-5),
        epochs=dnn_cfg.get("epochs", 400),
        lr=dnn_cfg.get("lr", 1e-3),
        batch_size=dnn_cfg.get("batch_size", 512),
        seed=dnn_cfg.get("seed", 1),
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

    print(f"DNN training — {n_batches} batches")
    print(f"Checkpoints → {_CKPT_DIR}\n")

    for batch_idx in range(n_batches):
        print(f"\n{'─' * 60}")
        print(f"Batch {batch_idx:02d} / {n_batches - 1}")
        train_batch(batch_idx, cfg)

    print("\nDNN training complete.")


if __name__ == "__main__":
    main()