"""
main.py — Entry point for the full pipeline.

Reproduces and extends Krauss, Do & Huck (2016) — "Deep Neural Networks,
Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500."

Pipeline stages (in order):
    1. Download  — Pull S&P 500 constituent history and daily returns from
                   WRDS/CRSP. Saves to data/raw/. Skipped if raw files exist.

    2. Processing — Clean returns, build constituent matrix, build valid
                    universe mask. Saves to data/processed/. Skipped if
                    processed files exist.

    3. Features  — Build 31 hand-crafted lag features per Section 4.2 of the
                   paper for DNN, GBT, and RAF models. Saves 23 batches of
                   (X_train, y_train, X_trade, y_trade, meta_trade) to
                   data/processed/features/. Skipped if batch files exist.

    4. Sequences — Build raw return sequences for LSTM and CNN extensions.
                   Same 23-batch structure as features. Saves to
                   data/processed/features/ with seq_ prefix. Skipped if
                   batch files exist.

    5. Train     — (optional, controlled by --skip-training / --train-only)
                   For each (model, batch) pair: load data, train model, and
                   save checkpoints to src/train/checkpoints/<model>/batch_<i>/.
                   If training is disabled, the pipeline validates that the
                   required checkpoints exist and raises an error if any are
                   missing.

    6. Inference — Load saved checkpoints for each (model, batch) pair, run
                   predict_proba on the trading window, and save per-batch
                   prediction parquets to results/. Skipped for any batch
                   whose prediction file already exists (use --force-inference
                   to re-run). After all base models are scored, two ensemble
                   variants are constructed:
                     • ensemble_base — equal-weighted average of DNN + GBT + RAF
                     • ensemble_seq  — equal-weighted average of LSTM + CNN

    7. Backtest  — Pass concatenated predictions to the backtest engine, which
                   handles portfolio construction (long top-k, short bottom-k),
                   transaction costs (0.05% per half-turn per Avellaneda &
                   Lee, 2010), and computes performance/risk metrics matching
                   Tables 1–4 in the paper.

    8. Results   — Save per-model daily return series, equity curves, and
                   summary metrics to results/. Print a final comparison table
                   to stdout in the same format as Table 2 of the paper.

Skip / resume logic:
    Stages 1–4 check for their expected output files before running. Stage 6
    skips any batch whose prediction parquet already exists. This means a run
    interrupted mid-inference can be resumed without reprocessing anything.

Usage:
    python main.py                               # full pipeline, all models
    python main.py --models lstm cnn             # specific models only
    python main.py --batch 5                     # single batch (debugging)
    python main.py --skip-download               # assume raw data exists
    python main.py --skip-training               # inference + backtest only
    python main.py --train-only                  # stop after training
    python main.py --force-features              # re-run feature engineering
    python main.py --force-inference             # re-run inference even if outputs exist

Ensembles:
    Two ensemble variants are always produced when all component models are run:
      ensemble_base — replicates the ENS result from the paper (DNN + GBT + RAF)
      ensemble_seq  — our novel LSTM + CNN ensemble

    To run a specific ensemble without rerunning base models (if their
    prediction files already exist):
      python main.py --models ensemble_base
      python main.py --models ensemble_seq

Environment variables:
    WRDS_USERNAME  — required for Stage 1
    SEQUENCE_LENGTH — override sequence length for LSTM/CNN (default: 240)

Checkpoint locations:
    src/train/checkpoints/<model_name>/batch_<ii>/
      PyTorch models (dnn, lstm, cnn): final.pt  (or latest.pt fallback)
      Sklearn/XGBoost (gbt, rf):       final.joblib  (or latest.joblib fallback)
"""
import os
import sys
import yaml
import random
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

from src.processing.download_wrds import (connect_wrds,
                                          download_constituents,
                                          download_returns)
from src.processing.data_processing import main as run_processing
from src.processing.feature_engineering import main as run_features
from src.processing.sequence_engineering import main as run_sequences
from src.train.train_lstm import main as lstm_train
from src.train.train_cnn import main as cnn_train
from src.train.train_gbt import main as gbt_train
from src.train.train_rf import main as rf_train
from src.train.train_dnn import main as dnn_train
from src.inference.run_inference import run_inference
from src.backtest.backtest import (run_backtest_from_predictions,
                                   compute_metrics,
                                   compute_metrics_by_subperiod,
                                   save_results)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Base models that have their own checkpoints + training pipelines
BASE_MODELS = ["dnn", "random_forest", "lstm", "cnn"]
# Ensemble labels — built from base model predictions, no separate training
ENSEMBLE_MODELS = ["ensemble_base", "ensemble_seq"]
ALL_MODELS = BASE_MODELS + ENSEMBLE_MODELS

# Checkpoint extension by model family
PT_MODELS = {"dnn", "lstm", "cnn"}       # PyTorch — saved as .pt
JOBLIB_MODELS = {"gbt", "random_forest"}     # sklearn/XGBoost — saved as .joblib

def load_config() -> dict:
    with open(ROOT/"config"/"config.yaml") as f:
        return yaml.safe_load(f)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StatArb Pipeline — Krauss et al. (2016) Reproduction & Extension",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="+",
                        choices=ALL_MODELS,
                        default=None,
                        help=("Models to run. Choices: dnn gbt random_forest lstm cnn "
                              "ensemble_base ensemble_seq. "
                              "Default: all models enabled in config.yaml."),)
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Run a single batch index (0–22). Useful for debugging.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip WRDS download. Raw data must already exist.",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help=(
            "Skip the training stage entirely. Inference will load existing "
            "checkpoints from src/train/checkpoints/. Raises an error if any "
            "required checkpoint is missing."
        ),
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Run training only — stop before inference and backtest.",
    )
    parser.add_argument(
        "--force-features", action="store_true",
        help="Re-run feature and sequence engineering even if files exist.",
    )
    parser.add_argument(
        "--force-inference", action="store_true",
        help="Re-run inference even if prediction files already exist.",
    )
    return parser.parse_args()

def all_batch_files_exist(directory: Path, prefix: str, n_batches: int) -> bool:
    suffixes = ["X_train", "y_train", "X_trade", "y_trade", "meta_trade"]
    for i in range(1, n_batches + 1):
        for s in suffixes:
            p = directory / f"{prefix}_{i:02d}_{s}.parquet"
            if not p.exists() or p.stat().st_size == 0:
                return False
    return True

def _checkpoint_path(ckpt_root: Path, model_name: str, batch_idx: int) -> Path:
    """Return the checkpoint directory for a given model and batch."""
    return ckpt_root/model_name/f"batch_{batch_idx:02d}"

def _checkpoint_exists(model_name: str, batch_idx: int, ckpt_root: Path) -> bool:
    """
    Return True if a usable checkpoint exists for this (model, batch) pair.

    Accepts final.pt / latest.pt for PyTorch models and
    final.joblib / latest.joblib for sklearn/XGBoost models.
    """
    ckpt_dir = _checkpoint_path(ckpt_root, model_name, batch_idx)
    if model_name in PT_MODELS:
        return (ckpt_dir / "final.pt").exists() or (ckpt_dir / "latest.pt").exists()
    elif model_name in JOBLIB_MODELS:
        return (
            (ckpt_dir / "final.joblib").exists()
            or (ckpt_dir / "latest.joblib").exists()
        )
    return False  # ensemble models have no checkpoints of their own

def validate_checkpoints(
    base_models: list[str],
    batches: list[int],
    ckpt_root: Path,
) -> None:
    """
    Check that every required checkpoint exists when training is skipped.

    Raises RuntimeError listing ALL missing checkpoints (not just the first)
    so the user can see the complete picture before re-running training.
    """
    missing = []
    for model_name in base_models:
        if model_name in ENSEMBLE_MODELS:
            continue  # ensembles are built from predictions, not checkpoints
        for batch_idx in batches:
            if not _checkpoint_exists(model_name, batch_idx, ckpt_root):
                ckpt_dir = _checkpoint_path(ckpt_root, model_name, batch_idx)
                missing.append(f"  {model_name} / batch_{batch_idx:02d}  →  {ckpt_dir}")

    if missing:
        bullet_list = "\n".join(missing)
        raise RuntimeError(
            "--skip-training was set, but the following checkpoints are missing:\n"
            f"{bullet_list}\n\n"
            "Re-run without --skip-training to train the missing models, or point "
            "--models to only the models whose checkpoints exist."
        )
    print("Checkpoint validation passed — all required checkpoints found.")

def load_returns(proc_dir: Path, cfg: dict) -> pd.DataFrame:
    """
    Load returns_clean.parquet — wide matrix (index=date, columns=permno).
    Required by the backtest engine to attach next-day realised returns.
    """
    path = proc_dir / cfg["data"]["returns_clean_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"returns_clean not found at {path}. Run the processing stage first."
        )
    ret = pd.read_parquet(path)
    ret.index = pd.to_datetime(ret.index)
    ret.columns = ret.columns.astype(int)
    return ret

def load_predictions_for_model(
    res_dir: Path,
    model_name: str,
    batches: list[int],
) -> pd.DataFrame | None:
    """
    Concatenate per-batch prediction parquets saved during Stage 6.
    Returns None if no prediction files exist for this model.

    Expected filename: results/<model_name>_batch_<ii>_predictions.parquet
    Columns: date, permno, score
    """
    frames = []
    for i in batches:
        p = res_dir / f"{model_name}_batch_{i:02d}_predictions.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            print(f"  [WARN] Missing prediction file: {p.name} — skipping batch {i}.")

    if not frames:
        print(f"  [SKIP] No prediction files found for {model_name}.")
        return None

    df = pd.concat(frames, ignore_index=True)
    df["date"]   = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def print_summary_table(metrics_results: dict) -> None:
    """
    Print a comparison table matching Table 2 of Krauss et al. (2016).
    Rows = models, columns = key annualised risk/return metrics.
    """
    if not metrics_results:
        return

    rows = []
    for model_name, m in metrics_results.items():
        rows.append({
            "Model":              model_name.upper(),
            "Avg Daily Ret (%)":  f"{m['avg_daily_return'] * 100:.4f}",
            "Ann. Return (%)":    f"{m['annualized_return'] * 100:.2f}",
            "Ann. Sharpe":        f"{m.get('annualized_sharpe', m.get('sharpe_ratio', float('nan'))):.4f}",
            "Max Drawdown (%)":   f"{m['max_drawdown'] * 100:.2f}",
            "Win Rate (%)":       f"{m['win_rate'] * 100:.2f}",
            "Num Days":           str(m["num_days"]),
        })

    df = pd.DataFrame(rows).set_index("Model")
    print("\n" + "═" * 80)
    print("PERFORMANCE SUMMARY  (net of 0.05% per half-turn transaction costs)")
    print("═" * 80)
    print(df.to_string())
    print("═" * 80 + "\n")


def main():
    args = parse_args()
    cfg = load_config()

    # Validate incompatible flags
    if args.skip_training and args.train_only:
        print("ERROR: --skip-training and --train-only are mutually exclusive.")
        sys.exit(1)

    random.seed(cfg["reproducibility"]["global_seed"])
    np.random.seed(cfg["reproducibility"]["global_seed"])

    raw_dir = ROOT/cfg["data"]["raw_dir"]
    proc_dir = ROOT/cfg["data"]["processed_dir"]
    feat_dir = ROOT/cfg["data"]["features_dir"]
    res_dir = ROOT/cfg["data"]["results_dir"]
    ckpt_root = ROOT/"src"/"train"/"checkpoints"

    res_dir.mkdir(parents=True, exist_ok=True)

    n_batches = cfg["windows"]["n_batches"]

    # Determine which models and batches to process
    models_to_run = args.models or [
        m for m in ALL_MODELS if cfg["models"].get(m, {}).get("enabled", False)
    ]
    batches_to_run = [args.batch] if args.batch is not None else list(range(1, n_batches + 1))

    # Base models are those that need checkpoint loading (not pure ensembles)
    base_models_to_run = [m for m in models_to_run if m in BASE_MODELS]

    print(f"Models  : {models_to_run}")
    print(f"Batches : {batches_to_run}")
    print(f"Training: {'DISABLED (--skip-training)' if args.skip_training else 'ENABLED'}")

    # Stage 1 — Download
    constituents_path = raw_dir / cfg["data"]["constituents_file"]
    returns_path = raw_dir / cfg["data"]["returns_file"]

    if constituents_path.exists() and returns_path.exists():
        print("\nStage 1: Raw data exists — skipping download.")
    elif args.skip_download:
        print("ERROR: --skip-download set but raw data not found. Exiting.")
        sys.exit(1)
    else:
        print("\nStage 1: Downloading from WRDS …")
        raw_dir.mkdir(parents=True, exist_ok=True)
        db = connect_wrds()
        constituents_df = download_constituents(
            db, start=cfg["data"]["start_date"], end=cfg["data"]["end_date"]
        )
        download_returns(
            db, constituents=constituents_df,
            start=cfg["data"]["start_date"], end=cfg["data"]["end_date"]
        )
        db.close()

    # Stage 2 — Processing
    processed_files = [
        proc_dir / cfg["data"]["returns_clean_file"],
        proc_dir / cfg["data"]["constituent_matrix_file"],
        proc_dir / cfg["data"]["valid_universe_file"],
    ]
    if all(f.exists() for f in processed_files):
        print("Stage 2: Processed data exists — skipping processing.")
    else:
        print("\nStage 2: Running data processing …")
        run_processing()

    # Stage 3 — Feature engineering
    if not args.force_features and all_batch_files_exist(feat_dir, "batch", n_batches):
        print("Stage 3: Feature batches exist — skipping feature engineering.")
    else:
        print("\nStage 3: Running feature engineering …")
        run_features()

    # Stage 4 — Sequence engineering
    needs_sequences = any(m in ("lstm", "cnn", "ensemble_seq") for m in models_to_run)
    if needs_sequences:
        if not args.force_features and all_batch_files_exist(
            feat_dir, "seq_batch", n_batches
        ):
            print("Stage 4: Sequence batches exist — skipping sequence engineering.")
        else:
            print("\nStage 4: Running sequence engineering …")
            os.environ["SEQUENCE_LENGTH"] = str(cfg["sequences"]["sequence_length"])
            run_sequences()
    else:
        print("Stage 4: No sequence models requested — skipping.")

    # Stage 5 — Training
    print(f"\n{'═' * 60}")
    if args.skip_training:
        # Validate that all needed checkpoints already exist
        print("Stage 5: Training SKIPPED — validating existing checkpoints …")
        validate_checkpoints(base_models_to_run, batches_to_run, ckpt_root)
    else:
        print("Stage 5: Training …")

        # Map model name → training function for the two PyTorch models that
        # have dedicated train scripts. DNN / GBT / RAF training is expected
        # to be triggered via their own train scripts (train_dnn.py, etc.) or
        # wired here similarly to lstm/cnn.
        seq_trainers = {
            "lstm": lstm_train,
            "cnn":  cnn_train,
            "gbt": gbt_train,
            "dnn": dnn_train,
            "rf": rf_train,
        }

        for model_name in base_models_to_run:
            if model_name not in seq_trainers:
                print(
                    f"  [INFO] Training for {model_name} is handled via its own "
                    "train script (not wired here). If checkpoints are missing, "
                    "run the appropriate train script first."
                )
                continue

            print(f"\n{'─' * 60}")
            print(f"Training: {model_name.upper()}")
            seq_trainers[model_name]()

    if args.train_only:
        print("\n--train-only set. Stopping after training.")
        return

    # Stage 6 — Inference
    print(f"\n{'═' * 60}")
    print("Stage 6: Inference …")

    run_inference(
        models_to_run=models_to_run,
        batches=batches_to_run,
        ckpt_root=ckpt_root,
        feat_dir=feat_dir,
        res_dir=res_dir,
        cfg=cfg,
        force=args.force_inference,
    )

    # Stage 7 — Backtest
    print(f"\n{'═' * 60}")
    print("Stage 7: Backtest …")

    k = cfg["trading"]["k"]
    transaction_cost = cfg["trading"]["transaction_cost_per_half_turn"]
    returns = load_returns(proc_dir, cfg)

    # Include ensemble variants in backtesting
    all_models_for_backtest = list(models_to_run)
    if "ensemble_base" not in all_models_for_backtest:
        # Auto-include ensemble_base if all components ran
        if {"dnn", "gbt", "random_forest"}.issubset(set(models_to_run)):
            all_models_for_backtest.append("ensemble_base")
    if "ensemble_seq" not in all_models_for_backtest:
        if {"lstm", "cnn"}.issubset(set(models_to_run)):
            all_models_for_backtest.append("ensemble_seq")

    backtest_results = {}
    metrics_results = {}
    subperiod_results = {}

    for model_name in all_models_for_backtest:
        print(f"\n{'─' * 40}")
        print(f"Backtesting: {model_name.upper()}")

        predictions = load_predictions_for_model(res_dir, model_name, batches_to_run)
        if predictions is None:
            continue

        try:
            daily = run_backtest_from_predictions(
                predictions=predictions,
                returns=returns,
                k=k,
                transaction_cost=transaction_cost,
            )
        except ValueError as e:
            print(f"  [ERROR] Backtest failed for {model_name}: {e}")
            continue

        metrics = compute_metrics(daily, return_col="net_return")
        subperiods = compute_metrics_by_subperiod(daily, return_col="net_return")

        backtest_results[model_name] = daily
        metrics_results[model_name] = metrics
        subperiod_results[model_name] = subperiods

        save_results(
            daily_results=daily,
            metrics=metrics,
            model_name=model_name,
            output_dir=str(res_dir),
        )

        sharpe = metrics.get("annualized_sharpe", metrics.get("sharpe_ratio", float("nan")))
        print(f"  Annualised Sharpe : {sharpe:.4f}")
        print(f"  Avg daily return  : {metrics['avg_daily_return']:.4%}")
        print(f"  Max drawdown      : {metrics['max_drawdown']:.4%}")
        print(f"  Win rate          : {metrics['win_rate']:.4%}")
        print(f"  Num trading days  : {metrics['num_days']}")

    # Stage 8 — Results
    print_summary_table(metrics_results)

    if metrics_results:
        summary_df = pd.DataFrame(metrics_results).T
        summary_df.index.name = "model"
        summary_df.to_parquet(res_dir / "summary.parquet")
        summary_df.to_csv(res_dir / "summary.csv")
        print(f"Summary saved → {res_dir / 'summary.csv'}")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()