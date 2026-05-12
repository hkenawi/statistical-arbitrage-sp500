"""
src/inference/run_inference.py — Load saved model checkpoints and run inference.

For each (model, batch) pair:
  1. Load the appropriate checkpoint from src/train/checkpoints/<model>/batch_<i>/
  2. Load the matching feature/sequence batch from data/processed/features/
  3. Call model.predict_proba(X_trade) and save predictions to results/

Supports all five base models (dnn, gbt, random_forest, lstm, cnn) and two
ensemble variants:
  - ensemble_base : equal-weighted average of DNN + GBT + RAF predictions
  - ensemble_seq  : equal-weighted average of LSTM + CNN predictions

Checkpoint format expected per model:
  - PyTorch models (dnn, lstm, cnn):   <checkpoint_dir>/final.pt
  - Sklearn/XGBoost (gbt, rf):         <checkpoint_dir>/final.joblib

If `final.pt` / `final.joblib` is missing, falls back to `latest.pt` /
`latest.joblib` so that a partially-completed training run can still be
evaluated.

Output per batch:
  results/<model_name>_batch_<ii>_predictions.parquet
  columns: date, permno, score
"""

import numpy as np
import pandas as pd
import torch
import joblib

from pathlib import Path


# ─── Model class imports ──────────────────────────────────────────────────────

from src.models.dnn import DNNModel, _DNNNetwork
from src.models.gbt import GBTModel
from src.models.random_forest import RandomForestModel
from src.models.lstm import LSTMModel, _LSTMNetwork
from src.models.cnn import CNNModel, _CNNNetwork

# Helpers

def _pt_path(ckpt_dir: Path) -> Path:
    """Return final.pt if it exists, else latest.pt. Raise if neither found."""
    for name in ("final.pt", "latest.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. "
        "Expected final.pt or latest.pt. Has the model been trained?"
    )


def _joblib_path(ckpt_dir: Path) -> Path:
    """Return final.joblib if it exists, else latest.joblib. Raise if neither."""
    for name in ("final.joblib", "latest.joblib"):
        p = ckpt_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. "
        "Expected final.joblib or latest.joblib. Has the model been trained?"
    )

# Model loaders

def _load_dnn(ckpt_dir: Path, cfg: dict) -> DNNModel:
    """Reconstruct DNNModel and load weights from checkpoint."""
    arch = cfg["models"]["dnn"].get("architecture", [31, 31, 10, 5, 2])
    dropout_hidden = cfg["models"]["dnn"].get("dropout_hidden", 0.5)
    dropout_input = cfg["models"]["dnn"].get("dropout_input", 0.1)

    model = DNNModel(architecture=arch,
                     dropout_hidden=dropout_hidden,
                     dropout_input=dropout_input)
    model.network = _DNNNetwork(arch, dropout_hidden, dropout_input)
    model.network.load_state_dict(
        torch.load(_pt_path(ckpt_dir), map_location="cpu", weights_only=True)
    )
    model.network.eval()
    return model


def _load_lstm(ckpt_dir: Path, cfg: dict) -> LSTMModel:
    """Reconstruct LSTMModel and load weights from checkpoint."""
    mcfg = cfg["models"].get("lstm", {})
    hidden_size = mcfg.get("hidden_size", 64)
    n_layers    = mcfg.get("n_layers", 2)
    dropout     = mcfg.get("dropout", 0.3)

    model = LSTMModel(hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
    model.network = _LSTMNetwork(hidden_size, n_layers, dropout)
    model.network.load_state_dict(
        torch.load(_pt_path(ckpt_dir), map_location="cpu", weights_only=True)
    )
    model.network.eval()
    return model


def _load_cnn(ckpt_dir: Path, seq_len: int, cfg: dict) -> CNNModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = cfg["models"].get("cnn", {})
    num_filters = mcfg.get("num_filters", 64)
    kernel_size = mcfg.get("kernel_size", 5)
    num_layers = mcfg.get("num_layers", 2)
    dropout = mcfg.get("dropout", 0.3)

    model = CNNModel(num_filters=num_filters, kernel_size=kernel_size,
                     num_layers=num_layers, dropout=dropout, device=device)
    model.seq_len = seq_len
    model.network = _CNNNetwork(seq_len, num_filters, kernel_size, num_layers, dropout).to(device)
    model.network.load_state_dict(
        torch.load(_pt_path(ckpt_dir), map_location=device, weights_only=True)
    )
    model.network.eval()
    return model


def _load_gbt(ckpt_dir: Path) -> GBTModel:
    """Load GBTModel from joblib checkpoint."""
    wrapper = GBTModel()
    wrapper.model = joblib.load(_joblib_path(ckpt_dir))
    return wrapper


def _load_rf(ckpt_dir: Path) -> RandomForestModel:
    """Load RandomForestModel from joblib checkpoint."""
    wrapper = RandomForestModel()
    wrapper.model = joblib.load(_joblib_path(ckpt_dir))
    return wrapper


# Feature / sequence loaders

def _load_feature_batch(feat_dir: Path, batch_idx: int):
    """Load (X_trade, meta_trade) for lag-feature models (DNN / GBT / RAF)."""
    prefix = f"batch_{batch_idx:02d}"
    X_trade    = pd.read_parquet(feat_dir / f"{prefix}_X_trade.parquet")
    meta_trade = pd.read_parquet(feat_dir / f"{prefix}_meta_trade.parquet")
    return X_trade, meta_trade


def _load_seq_batch(feat_dir: Path, batch_idx: int):
    """Load (X_trade, meta_trade) for sequence models (LSTM / CNN)."""
    prefix = f"seq_batch_{batch_idx:02d}"
    X_trade    = pd.read_parquet(feat_dir / f"{prefix}_X_trade.parquet")
    meta_trade = pd.read_parquet(feat_dir / f"{prefix}_meta_trade.parquet")
    return X_trade, meta_trade


# Core inference routine

def _run_model_inference(
    model_name: str,
    batches: list[int],
    ckpt_root: Path,
    feat_dir: Path,
    res_dir: Path,
    cfg: dict,
    force: bool = False,
) -> None:
    """
    Run inference for one base model across all requested batches.

    Skips a batch if its prediction file already exists and force=False.
    Saves results/<model_name>_batch_<ii>_predictions.parquet per batch.
    """
    is_seq_model = model_name in ("lstm", "cnn")

    print(f"\n{'─' * 60}")
    print(f"Inference: {model_name.upper()}")

    for batch_idx in batches:
        out_path = res_dir / f"{model_name}_batch_{batch_idx:02d}_predictions.parquet"
        if out_path.exists() and not force:
            print(f"  [SKIP] Batch {batch_idx:02d} — predictions already exist.")
            continue

        ckpt_dir = ckpt_root/model_name/f"batch_{batch_idx:02d}"

        # Load checkpoint
        try:
            if is_seq_model:
                X_trade, meta_trade = _load_seq_batch(feat_dir, batch_idx)
                seq_len = X_trade.shape[1]
                if model_name == "lstm":
                    model = _load_lstm(ckpt_dir, cfg)
                else:
                    model = _load_cnn(ckpt_dir, seq_len, cfg)
            elif model_name == "dnn":
                X_trade, meta_trade = _load_feature_batch(feat_dir, batch_idx)
                model = _load_dnn(ckpt_dir, cfg)
            elif model_name == "gbt":
                X_trade, meta_trade = _load_feature_batch(feat_dir, batch_idx)
                model = _load_gbt(ckpt_dir)
            elif model_name == "random_forest":
                X_trade, meta_trade = _load_feature_batch(feat_dir, batch_idx)
                model = _load_rf(ckpt_dir)
            else:
                raise ValueError(f"Unknown model: {model_name}")

        except FileNotFoundError as e:
            print(f"  [ERROR] Batch {batch_idx:02d} — {e}")
            continue

        # Inference
        print(f"  Batch {batch_idx:02d} — running predict_proba on "
              f"{len(X_trade):,} observations …")
        scores = model.predict_proba(X_trade)

        # Save
        pred_df = meta_trade[["date", "permno"]].copy()
        pred_df["score"] = scores
        pred_df["date"]   = pd.to_datetime(pred_df["date"])
        pred_df["permno"] = pred_df["permno"].astype(int)
        pred_df.to_parquet(out_path, index=False)
        print(f"  [DONE]  Batch {batch_idx:02d} → {out_path.name}")


# Ensemble builders

def _build_ensemble(
    component_names: list[str],
    ensemble_name: str,
    batches: list[int],
    res_dir: Path,
    force: bool = False,
) -> None:
    """
    Average the scores of component models to produce ensemble predictions.

    Reads already-saved per-batch prediction parquets for each component,
    merges on (date, permno), and saves the averaged score as a new parquet.

    Skips any batch where the ensemble file already exists (unless force=True).
    Skips any batch where one or more component files are missing.
    """
    print(f"\n{'─' * 60}")
    print(f"Building ensemble: {ensemble_name.upper()} "
          f"← {' + '.join(c.upper() for c in component_names)}")

    for batch_idx in batches:
        out_path = res_dir / f"{ensemble_name}_batch_{batch_idx:02d}_predictions.parquet"
        if out_path.exists() and not force:
            print(f"  [SKIP] Batch {batch_idx:02d} — ensemble predictions already exist.")
            continue

        # Load component predictions
        frames = {}
        missing = False
        for name in component_names:
            p = res_dir / f"{name}_batch_{batch_idx:02d}_predictions.parquet"
            if not p.exists():
                print(f"  [WARN] Batch {batch_idx:02d} — missing component: {name}. "
                      f"Skipping batch.")
                missing = True
                break
            df = pd.read_parquet(p)
            df["date"]   = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype(int)
            frames[name] = df.set_index(["date", "permno"])["score"]

        if missing:
            continue

        # Average scores
        combined = pd.concat(frames.values(), axis=1)
        combined.columns = component_names

        # Drop rows where any component has NaN — should not happen in practice
        n_before = len(combined)
        combined = combined.dropna()
        if len(combined) < n_before:
            print(f"  [WARN] Batch {batch_idx:02d} — dropped "
                  f"{n_before - len(combined)} rows with NaN scores.")

        combined["score"] = combined[component_names].mean(axis=1)
        result = combined[["score"]].reset_index()   # date, permno, score

        result.to_parquet(out_path, index=False)
        print(f"  [DONE]  Batch {batch_idx:02d} → {out_path.name} "
              f"({len(result):,} observations)")


# Public entry point

def run_inference(
    models_to_run: list[str],
    batches: list[int],
    ckpt_root: Path,
    feat_dir: Path,
    res_dir: Path,
    cfg: dict,
    force: bool = False,
) -> None:
    """
    Run inference for all requested models and batches, then build ensembles.

    Parameters
    ----------
    models_to_run : list of model names from
                    {dnn, gbt, random_forest, lstm, cnn,
                     ensemble_base, ensemble_seq}
    batches       : list of batch indices (0-22)
    ckpt_root     : root directory containing per-model checkpoint subdirs,
                    e.g. src/train/checkpoints/
    feat_dir      : directory containing feature/sequence parquets
    res_dir       : directory to write prediction parquets
    cfg           : parsed config.yaml dict
    force         : if True, re-run inference even if output files exist
    """
    res_dir.mkdir(parents=True, exist_ok=True)

    # Separate base models from ensemble labels
    BASE_MODELS   = {"dnn", "gbt", "random_forest", "lstm", "cnn"}
    ENSEMBLE_BASE = "ensemble_base"  # DNN + GBT + RAF
    ENSEMBLE_SEQ  = "ensemble_seq"   # LSTM + CNN

    base_to_run     = [m for m in models_to_run if m in BASE_MODELS]
    ensemble_to_run = [m for m in models_to_run
                       if m in (ENSEMBLE_BASE, ENSEMBLE_SEQ)]

    # ── Base model inference ──────────────────────────────────────────────────
    for model_name in base_to_run:
        _run_model_inference(
            model_name=model_name,
            batches=batches,
            ckpt_root=ckpt_root,
            feat_dir=feat_dir,
            res_dir=res_dir,
            cfg=cfg,
            force=force,
        )

    # Ensemble construction
    # Auto-build ensembles if explicitly requested, or if all components ran
    requested_or_available = set(base_to_run) | set(ensemble_to_run)

    if ENSEMBLE_BASE in ensemble_to_run or (
        {"dnn", "gbt", "random_forest"}.issubset(requested_or_available)
        and ENSEMBLE_BASE not in models_to_run
    ):
        if ENSEMBLE_BASE in models_to_run or {"dnn", "gbt", "random_forest"}.issubset(
            set(models_to_run)
        ):
            _build_ensemble(
                component_names=["dnn", "gbt", "random_forest"],
                ensemble_name=ENSEMBLE_BASE,
                batches=batches,
                res_dir=res_dir,
                force=force,
            )

    if ENSEMBLE_SEQ in ensemble_to_run or (
        {"lstm", "cnn"}.issubset(requested_or_available)
        and ENSEMBLE_SEQ not in models_to_run
    ):
        if ENSEMBLE_SEQ in models_to_run or {"lstm", "cnn"}.issubset(set(models_to_run)):
            _build_ensemble(
                component_names=["lstm", "cnn"],
                ensemble_name=ENSEMBLE_SEQ,
                batches=batches,
                res_dir=res_dir,
                force=force,
            )