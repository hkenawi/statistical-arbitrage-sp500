"""
main.py — Entry point for the full pipeline.

Reproduces and extends Krauss, Do & Huck (2016) — "Deep Neural Networks,
Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500."

Pipeline stages (in order):
    1. Download — Pull S&P 500 constituent history and daily returns from
                  WRDS/CRSP. Saves to data/raw/. Skipped if raw files exist.

    2. Processing — Clean returns, build constituent matrix, build valid
                    universe mask. Saves to data/processed/. Skipped if
                    processed files exist.

    3. Features — Build 31 hand-crafted lag features per Section 4.2 of
                  the paper for DNN, GBT, and RAF models. Saves 23 batches
                  of (X_train, y_train, X_trade, y_trade, meta_trade) to
                  data/processed/features/. Skipped if batch files exist.

    4. Sequences — Build raw return sequences for LSTM and CNN extensions.
                   Same 23-batch structure as features. Saves to
                   data/processed/features/ with seq_ prefix. Skipped if
                   batch files exist.

    5. Train — For each (model, batch) pair: load data, train model on
               training window, run predict_proba on trading window.
               Saves per-batch prediction parquets to results/ so
               interrupted runs resume from where they left off.

    6. Backtest — Pass concatenated predictions to the backtest engine,
                  which handles portfolio construction (long top-k, short
                  bottom-k), transaction costs (0.05% per half-turn per
                  Avellaneda & Lee 2010), and performance/risk metrics.

    7. Results — Save summary performance table to results/summary.parquet
                 and results/summary.csv.

Skip logic:
    Each stage checks for its expected output files before running. If they
    exist the stage is skipped and the pipeline moves on.

Usage:
    python main.py                            # run full pipeline
    python main.py --models lstm cnn          # run specific models only
    python main.py --batch 5                  # run single batch (debugging)
    python main.py --skip-download            # skip WRDS download
    python main.py --force-features           # re-run feature/sequence engineering

Environment variables:
    WRDS_USERNAME — required for Stage 1
    SEQUENCE_LENGTH — override sequence length for LSTM/CNN (default: 240)
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
from src.backtest.backtest import (run_backtest_from_predictions,
                                   compute_metrics,
                                   compute_metrics_by_subperiod,
                                   save_results,)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)

def load_returns(proc_dir: Path, cfg: dict) -> pd.DataFrame:
    """
    Load returns_clean.parquet — wide matrix (index=date, columns=permno).
    Required by the backtest engine to attach next-day realized returns.
    """
    path = proc_dir / cfg["data"]["returns_clean_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"returns_clean not found at {path}. "
            "Run processing stage first."
        )
    ret = pd.read_parquet(path)
    ret.index = pd.to_datetime(ret.index)
    ret.columns = ret.columns.astype(int)
    return ret

def load_predictions_for_model(
    res_dir: Path,
    model_name: str,
    batches: list[int],
    n_batches: int,
) -> pd.DataFrame | None:
    """
    Concatenate per-batch prediction parquets saved during Stage 5.
    Returns None if no prediction files exist for this model yet.

    Expected file format per batch:
        results/{model_name}_batch_{i:02d}_predictions.parquet
        columns: date, permno, score
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StatArb Pipeline — Krauss et al. (2016) Reproduction & Extension"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["dnn", "gbt", "random_forest", "lstm", "cnn", "ensemble"],
        default=None,
        help="Models to run. Default: all enabled in config.yaml."
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Run a single batch index (0-22). Useful for debugging."
    )
    parser.add_argument(
        "--force-features", action="store_true",
        help="Re-run feature and sequence engineering even if files exist."
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip WRDS download. Requires raw data to already exist."
    )
    return parser.parse_args()

def all_batch_files_exist(directory: Path, prefix: str, n_batches: int) -> bool:
    suffixes = ["X_train", "y_train", "X_trade", "y_trade", "meta_trade"]
    for i in range(n_batches):
        for s in suffixes:
            p = directory / f"{prefix}_{i:02d}_{s}.parquet"
            if not p.exists() or p.stat().st_size == 0:
                return False
    return True


def main():
    args = parse_args()
    cfg = load_config()

    random.seed(cfg["reproducibility"]["global_seed"])
    np.random.seed(cfg["reproducibility"]["global_seed"])

    raw_dir = ROOT/cfg["data"]["raw_dir"]
    proc_dir = ROOT/cfg["data"]["processed_dir"]
    feat_dir = ROOT/cfg["data"]["features_dir"]
    res_dir = ROOT/cfg["data"]["results_dir"]
    res_dir.mkdir(parents=True, exist_ok=True)

    n_batches = cfg["windows"]["n_batches"]

    # Stage 1 — Download
    constituents_path = raw_dir/cfg["data"]["constituents_file"]
    returns_path = raw_dir/cfg["data"]["returns_file"]
    if constituents_path.exists() and returns_path.exists():
        print("Raw data exists — skipping download.")
    elif args.skip_download:
        print("--skip-download set but raw data not found. Exiting.")
        sys.exit(1)
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
        db = connect_wrds()
        constituents_df = download_constituents(db, start=cfg["data"]["start_date"], end=cfg["data"]["end_date"])
        download_returns(db, constituents=constituents_df, start=cfg["data"]["start_date"], end=cfg["data"]["end_date"])
        db.close()

    # Stage 2 — Processing
    processed_files = [proc_dir/cfg["data"]["returns_clean_file"],
                       proc_dir/cfg["data"]["constituent_matrix_file"],
                       proc_dir/cfg["data"]["valid_universe_file"],]
    if all(f.exists() for f in processed_files):
        print("Processed data exists — skipping processing.")
    else:
        run_processing()

    # Stage 3 — Feature engineering (31-lag features for DNN/GBT/RAF)
    if not args.force_features and all_batch_files_exist(feat_dir, "batch", n_batches):
        print("Feature batches exist — skipping feature engineering.")
    else:
        run_features()

    # Stage 4 — Sequence engineering (raw sequences for LSTM/CNN)
    if not args.force_features and all_batch_files_exist(feat_dir, "seq_batch", n_batches):
        print("Sequence batches exist — skipping sequence engineering.")
    else:
        os.environ["SEQUENCE_LENGTH"] = str(cfg["sequences"]["sequence_length"])
        run_sequences()

    # Stage 5 — Train
    all_models = ["dnn", "gbt", "random_forest", "lstm", "cnn", "ensemble"]
    models_to_run = args.models or [m for m in all_models if cfg["models"].get(m, {}).get("enabled", False)]
    batches_to_run = [args.batch] if args.batch is not None else list(range(n_batches))

    print(f"Models  : {models_to_run}")
    print(f"Batches : {batches_to_run}")

    seq_trainers = {
        "lstm": lstm_train,
        "cnn":  cnn_train,
    }

    for model_name in models_to_run:
        if model_name not in seq_trainers:
            print(f"Training for {model_name} not yet wired into pipeline — skipping.")
            continue

        print(f"\n{'═' * 60}")
        print(f"{model_name.upper()} training")
        seq_trainers[model_name]()

        # Stage 6 — Backtest
        k = cfg["trading"]["k"]
        transaction_cost = cfg["trading"]["transaction_cost_per_half_turn"]

        returns = load_returns(proc_dir, cfg)

        backtest_results = {}  # model_name -> daily_results DataFrame
        metrics_results = {}  # model_name -> metrics dict
        subperiod_results = {}  # model_name -> subperiod DataFrame

        for model_name in models_to_run:
            print(f"\n{'─' * 60}")
            print(f"Backtesting: {model_name.upper()}")

            predictions = load_predictions_for_model(
                res_dir, model_name, batches_to_run, n_batches
            )
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

            print(f"  Annualized Sharpe : {metrics['annualized_sharpe']:.4f}")
            print(f"  Avg daily return  : {metrics['avg_daily_return']:.4%}")
            print(f"  Max drawdown      : {metrics['max_drawdown']:.4%}")
            print(f"  Win rate          : {metrics['win_rate']:.4%}")
            print(f"  Num trading days  : {metrics['num_days']}")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()