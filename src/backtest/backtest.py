"""
Backtest engine for statistical arbitrage strategy.

Logic follows Krauss, Do & Huck:
- each day rank stocks by predicted P(outperform)
- long top k stocks
- short bottom k stocks
- equal weight both sides
- daily portfolio return = mean(long returns) - mean(short returns)
- apply transaction cost
"""

import pandas as pd
import numpy as np


def prepare_returns_long(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Convert returns matrix date x permno into long format:
    date | permno | next_return
    """
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)

    long_ret = (
        returns.stack()
        .reset_index()
    )

    long_ret.columns = ["date", "permno", "next_return"]
    return long_ret


def run_backtest(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    k: int = 10,
    transaction_cost: float = 0.0005,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    predictions:
        DataFrame with columns: date, permno, score
        score = predicted probability of outperforming next day

    returns:
        Either:
        1. wide matrix: date x permno
        2. long format: date, permno, next_return

    k:
        Number of stocks to long and short each day.

    transaction_cost:
        0.0005 = 0.05% per half-turn, same as paper.

    Returns
    -------
    daily_results:
        daily long return, short return, gross return, net return, cumulative return
    """

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])

    # Convert returns if wide format
    if "next_return" not in returns.columns:
        ret_long = prepare_returns_long(returns)
    else:
        ret_long = returns.copy()
        ret_long["date"] = pd.to_datetime(ret_long["date"])

    # Merge predictions with actual next-day returns
    df = pred.merge(ret_long, on=["date", "permno"], how="inner")
    df = df.dropna(subset=["score", "next_return"])

    daily_rows = []

    for date, group in df.groupby("date"):
        group = group.sort_values("score", ascending=False)

        if len(group) < 2 * k:
            continue

        long_stocks = group.head(k)
        short_stocks = group.tail(k)

        long_ret = long_stocks["next_return"].mean()
        short_ret = short_stocks["next_return"].mean()

        gross_ret = long_ret - short_ret

        # Daily rebalance: pay cost on long leg and short leg
        # Paper uses 0.05% per share per half-turn.
        # Simple version subtracts 2 * cost per day.
        net_ret = gross_ret - 2 * transaction_cost

        daily_rows.append({
            "date": date,
            "long_return": long_ret,
            "short_return": short_ret,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "n_stocks": len(group),
        })

    daily_results = pd.DataFrame(daily_rows).sort_values("date")
    daily_results["cum_gross_return"] = (1 + daily_results["gross_return"]).cumprod() - 1
    daily_results["cum_net_return"] = (1 + daily_results["net_return"]).cumprod() - 1

    return daily_results


def compute_metrics(daily_results: pd.DataFrame, return_col: str = "net_return") -> dict:
    """
    Compute basic performance metrics.
    """

    r = daily_results[return_col].dropna()

    if len(r) == 0:
        raise ValueError("No returns available to compute metrics.")

    avg_daily_return = r.mean()
    daily_vol = r.std()
    sharpe = np.sqrt(252) * avg_daily_return / daily_vol if daily_vol != 0 else np.nan

    cumulative_return = (1 + r).prod() - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(r)) - 1

    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    win_rate = (r > 0).mean()

    var_5 = r.quantile(0.05)
    cvar_5 = r[r <= var_5].mean()

    return {
        "avg_daily_return": avg_daily_return,
        "annualized_return": annualized_return,
        "daily_volatility": daily_vol,
        "sharpe_ratio": sharpe,
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "VaR_5pct": var_5,
        "CVaR_5pct": cvar_5,
        "num_days": len(r),
    }


def save_results(daily_results: pd.DataFrame, metrics: dict, model_name: str):
    """
    Save daily returns and metrics.
    """

    daily_path = f"results/{model_name}_daily_returns.csv"
    metrics_path = f"results/{model_name}_metrics.csv"

    daily_results.to_csv(daily_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"Saved daily results to {daily_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    """
    Example usage after teammates generate predictions:

    predictions = pd.read_csv("results/lstm_predictions.csv")
    returns = pd.read_parquet("data/processed/returns_clean.parquet")

    daily = run_backtest(predictions, returns, k=10)
    metrics = compute_metrics(daily)

    save_results(daily, metrics, model_name="lstm")
    print(metrics)
    """

    print("Backtest template ready. Import run_backtest() after predictions are generated.")
