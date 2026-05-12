"""
Shared utilities for the Streamlit dashboard.

Includes:
    - Data loaders (cached)
    - Color palette
    - Model label mapping
    - Paper benchmark results
    - Plotly layout theme
"""

import sys
import yaml
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Constants
COLORS = {
    "dnn":            "#4361ee",
    "gbt":            "#f72585",
    "random_forest":  "#4cc9f0",
    "lstm":           "#f77f00",
    "cnn":            "#a8dadc",
    "ensemble_base":  "#ffffff",
    "ensemble_seq":   "#fcbf49",
    "market":         "#6c757d",
}

MODEL_LABELS = {
    "dnn":            "DNN",
    "gbt":            "GBT",
    "random_forest":  "RAF",
    "lstm":           "LSTM",
    "cnn":            "CNN",
    "ensemble_base":  "ENS (Base)",
    "ensemble_seq":   "ENS (Seq)",
}

PAPER_RESULTS = {
    "DNN": {"avg_daily_return": 0.0013, "sharpe_ratio": 0.5521, "max_drawdown": -0.9544, "win_rate": 0.5174},
    "GBT": {"avg_daily_return": 0.0017, "sharpe_ratio": 1.2310, "max_drawdown": -0.8425, "win_rate": 0.5351},
    "RAF": {"avg_daily_return": 0.0023, "sharpe_ratio": 1.9008, "max_drawdown": -0.6689, "win_rate": 0.5423},
    "ENS": {"avg_daily_return": 0.0025, "sharpe_ratio": 1.8073, "max_drawdown": -0.7367, "win_rate": 0.5367},
}

PLOT_LAYOUT = dict(
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#0d0d18",
    font=dict(family="DM Sans", color="#e8e8f0", size=12),
    xaxis=dict(gridcolor="#1e1e2e", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e1e2e", showgrid=True, zeroline=False),
    legend=dict(bgcolor="#13131f", bordercolor="#1e1e2e", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
)

# Config
@st.cache_data
def load_config() -> dict:
    with open(ROOT/"config"/"config.yaml") as f:
        return yaml.safe_load(f)

# Data loaders
@st.cache_data
def load_returns() -> pd.DataFrame | None:
    cfg = load_config()
    path = ROOT / cfg["data"]["processed_dir"] / cfg["data"]["returns_clean_file"]
    if not path.exists():
        return None
    ret = pd.read_parquet(path)
    ret.index = pd.to_datetime(ret.index)
    ret.columns = ret.columns.astype(int)
    return ret


@st.cache_data
def load_daily_results(model_name: str) -> pd.DataFrame | None:
    cfg = load_config()
    path = ROOT / cfg["data"]["results_dir"] / f"{model_name}_daily_returns.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_metrics(model_name: str) -> dict | None:
    cfg = load_config()
    path = ROOT / cfg["data"]["results_dir"] / f"{model_name}_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).iloc[0].to_dict()


@st.cache_data
def load_all_predictions(model_name: str, n_batches: int = 22) -> pd.DataFrame | None:
    cfg = load_config()
    res_dir = ROOT / cfg["data"]["results_dir"]
    frames = []
    for i in range(1, n_batches + 1):
        p = res_dir / f"{model_name}_batch_{i:02d}_predictions.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def get_available_models() -> list[str]:
    cfg = load_config()
    res_dir = ROOT/cfg["data"]["results_dir"]
    all_models = ["dnn", "gbt", "random_forest", "lstm", "cnn", "ensemble_base", "ensemble_seq"]
    return [m for m in all_models if (res_dir / f"{m}_daily_returns.csv").exists()]