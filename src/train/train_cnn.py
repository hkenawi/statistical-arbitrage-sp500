"""CNN training."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from pathlib import Path

from src.models.cnn import CNNModel, _CNNNetwork

root = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(root / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(network: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    batch_idx: int,
                    val_loss: float,
                    path: Path) -> None:
    torch.save({
        "epoch": epoch,
        "batch": batch_idx,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)


def train_batch(batch_idx: int, cfg: dict, ckpt_dir: Path) -> None:
    seq_dir = root / cfg["data"]["seq_dir"]
    x_path = seq_dir / f"seq_batch_{batch_idx:02d}_X_train.parquet"
    y_path = seq_dir / f"seq_batch_{batch_idx:02d}_y_train.parquet"

    if not x_path.exists():
        print(f"  Batch {batch_idx:02d}: data not found, skipping.")
        return

    X_train = pd.read_parquet(x_path)
    y_train = pd.read_parquet(y_path).squeeze()

    cnn_cfg = cfg["models"]["cnn"]
    train_cfg = cfg["training"]

    model = CNNModel(
        num_filters  = cnn_cfg["num_filters"],
        kernel_size  = cnn_cfg["kernel_size"],
        num_layers   = cnn_cfg["num_layers"],
        dropout      = cnn_cfg["dropout"],
        lr           = cnn_cfg["lr"],
        batch_size   = cnn_cfg["batch_size"],
        epochs       = cnn_cfg["epochs"],
        use_tuner    = cnn_cfg["use_tuner"],
        n_trials     = cnn_cfg["n_trials"],
        val_fraction = cnn_cfg["val_fraction"],
    )

    X_np = X_train.values
    y_np = y_train.values
    model.seq_len = X_np.shape[1]

    if model.use_tuner:
        model.best_params = model.tune(X_train, y_train)
        num_filters = model.best_params["num_filters"]
        kernel_size = model.best_params["kernel_size"]
        num_layers  = model.best_params["num_layers"]
        dropout     = model.best_params["dropout"]
        lr          = model.best_params["lr"]
        batch_size  = model.best_params["batch_size"]
        epochs      = model.best_params["epochs"]
    else:
        num_filters = model.num_filters
        kernel_size = model.kernel_size
        num_layers  = model.num_layers
        dropout     = model.dropout
        lr          = model.lr
        batch_size  = model.batch_size
        epochs      = model.epochs

    network   = _CNNNetwork(model.seq_len, num_filters, kernel_size,
                            num_layers, dropout).to(model.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    criterion = nn.BCELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor   = train_cfg["scheduler"]["factor"],
        patience = train_cfg["scheduler"]["patience"],
    )

    train_loader, val_loader = model._build_loaders(
        X_np, y_np, batch_size, val_fraction=model.val_fraction
    )

    es_patience  = train_cfg["early_stopping"]["patience"]
    es_min_delta = train_cfg["early_stopping"]["min_delta"]
    best_val_loss    = float("inf")
    patience_counter = 0
    checkpoint_every = train_cfg["checkpoint_every"]

    print(f"  Training CNN batch {batch_idx:02d} — {epochs} epochs  "
          f"filters={num_filters}, kernel={kernel_size}, layers={num_layers}, "
          f"dropout={dropout}, lr={lr}, batch={batch_size}")

    last_epoch = epochs
    for epoch in range(1, epochs + 1):
        train_loss = model._train_epoch(network, train_loader, optimizer, criterion)
        val_loss = model._evaluate(network, val_loader, criterion)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

        if epoch % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"cnn_batch_{batch_idx:02d}_epoch_{epoch:03d}.pt"
            save_checkpoint(network, optimizer, epoch, batch_idx, val_loss, ckpt_path)
            print(f"    Checkpoint saved → {ckpt_path.name}")

        if val_loss < best_val_loss - es_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= es_patience:
                print(f"    Early stopping at epoch {epoch} "
                      f"(no improvement for {es_patience} epochs).")
                last_epoch = epoch
                break

    final_path = ckpt_dir / f"cnn_batch_{batch_idx:02d}_final.pt"
    save_checkpoint(network, optimizer, last_epoch, batch_idx, val_loss, final_path)
    print(f"  Final model saved → {final_path.name}")


def main() -> None:
    cfg = load_config()
    set_seed(cfg["reproducibility"]["global_seed"])

    ckpt_dir = root / cfg["data"]["results_dir"] / "checkpoints" / "cnn"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_batches = cfg["windows"]["n_batches"]

    print(f"CNN training — {n_batches} batches")
    print(f"Checkpoints → {ckpt_dir}\n")

    for batch_idx in range(n_batches):
        print(f"\n{'─' * 60}")
        print(f"Batch {batch_idx:02d} / {n_batches - 1}")
        train_batch(batch_idx, cfg, ckpt_dir)

    print("\nCNN training complete.")


if __name__ == "__main__":
    main()
