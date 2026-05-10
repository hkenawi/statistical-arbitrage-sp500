"""
Trains one CNNModel per sliding-window batch (23 total).
All training logic is encapsulated in CNNModel.fit() — this script
is responsible only for data loading, model instantiation, and
orchestrating the batch loop.

Outputs per batch:
    src/train/checkpoints/cnn/batch_{i:02d}/latest.pt     — saved every epoch
    src/train/checkpoints/cnn/batch_{i:02d}/epoch_NNN.pt  — every N epochs
    src/train/checkpoints/cnn/batch_{i:02d}/final.pt      — end of training

Standalone usage:
    python -m src.train.train_cnn
"""
import sys
import yaml
import numpy as np
import pandas as pd
import torch

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.models.cnn import CNNModel

_FEAT_DIR = _ROOT/"data"/"processed"/"features"
_CKPT_DIR = Path(__file__).resolve().parent/"checkpoints"/"cnn"

def load_config() -> dict:
    with open(_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_batch(batch_idx: int, cfg: dict) -> None:
    """Train CNN on one sliding-window batch."""
    x_path = _FEAT_DIR / f"seq_batch_{batch_idx:02d}_X_train.parquet"
    y_path = _FEAT_DIR / f"seq_batch_{batch_idx:02d}_y_train.parquet"

    if not x_path.exists():
        print(f"  Batch {batch_idx:02d}: data not found, skipping.")
        return

    X_train = pd.read_parquet(x_path)
    y_train = pd.read_parquet(y_path).squeeze()

    cnn_cfg = cfg["models"]["cnn"]

    model = CNNModel(num_filters=cnn_cfg["num_filters"],
                     kernel_size=cnn_cfg["kernel_size"],
                     num_layers=cnn_cfg["num_layers"],
                     dropout=cnn_cfg["dropout"],
                     lr=cnn_cfg["lr"],
                     batch_size=cnn_cfg["batch_size"],
                     epochs=cnn_cfg["epochs"],
                     use_tuner=cnn_cfg["use_tuner"],
                     n_trials=cnn_cfg["n_trials"],
                     val_fraction=cnn_cfg["val_fraction"],)

    ckpt_dir = _CKPT_DIR/f"batch_{batch_idx:02d}"

    model.fit(X_train,
              y_train,
              checkpoint_dir=ckpt_dir,
              checkpoint_every=cfg["training"]["checkpoint_every"],)

def main() -> None:
    cfg = load_config()
    set_seed(cfg["reproducibility"]["global_seed"])

    n_batches = cfg["windows"]["n_batches"]

    print(f"CNN training — {n_batches} batches")
    print(f"Checkpoints → {_CKPT_DIR}\n")

    for batch_idx in range(n_batches):
        print(f"\n{'─' * 60}")
        print(f"Batch {batch_idx:02d} / {n_batches - 1}")
        train_batch(batch_idx, cfg)

    print("\nCNN training complete.")


if __name__ == "__main__":
    main()