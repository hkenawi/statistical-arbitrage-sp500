"""
Backtest engine for the S&P 500 statistical arbitrage project.

Real project data format:
- returns_clean.parquet: wide matrix, index=date, columns=permno, values=daily returns
- batch_xx_meta_trade.parquet: columns = date, permno
- model output: predict_proba(X_trade) gives one score per row of meta_trade

Trading logic follows Krauss, Do & Huck:
- rank stocks by predicted P(outperform)
- long top k stocks
- short bottom k stocks
- equal weight
"""

from pathlib import Path
import numpy as np
import pandas as pd


def build_predictions_df(meta_trade: pd.DataFrame, scores) -> pd.DataFrame:
    """
    Combine meta_trade with model scores.

    meta_trade must have:
        date, permno

    scores must have same length as meta_trade.
    """

    pred = meta_trade.copy()

    if len(pred) != len(scores):
        raise ValueError(
            f"Length mismatch: meta_trade has {len(pred)} rows, "
            f"but scores has {len(scores)} rows."
        )

    pred["date"] = pd.to_datetime(pred["date"])
    pred["permno"] = pred["permno"].astype(int)
    pred["score"] = np.asarray(scores)

    return pred


def attach_next_returns(predictions: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Attach actual next-day returns from returns_clean.parquet.

    returns_clean format:
        index = date
        columns = permno
        values = return

    IMPORTANT:
    meta_trade date is day t.
    The label predicts return on day t+1.
    So we use returns.shift(-1).
    """

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["permno"] = pred["permno"].astype(int)

    ret = returns.copy()
    ret.index = pd.to_datetime(ret.index)
    ret.columns = ret.columns.astype(int)

    next_returns = ret.shift(-1)

    rows = []
    for date, group in pred.groupby("date"):
        if date not in next_returns.index:
            continue

        permnos = group["permno"].values
        available = [p for p in permnos if p in next_returns.columns]

        if len(available) == 0:
            continue

        r = next_returns.loc[date, available]

        temp = group[group["permno"].isin(available)].copy()
        temp["next_return"] = temp["permno"].map(r.to_dict())
        rows.append(temp)

    if not rows:
        raise ValueError("No matching dates/permnos between predictions and returns.")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["score", "next_return"])

    return out


def run_backtest_from_predictions(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    k: int = 10,
    transaction_cost: float = 0.0005,
) -> pd.DataFrame:
    """
    Run long-short backtest.

    predictions columns:
        date, permno, score

    returns:
        returns_clean wide matrix.

    transaction_cost:
        0.0005 = 0.05% per half-turn.
        We subtract 2 * transaction_cost each day for simple daily rebalance cost.
    """

    df = attach_next_returns(predictions, returns)

    daily_rows = []

    for date, group in df.groupby("date"):
        group = group.dropna(subset=["score", "next_return"])
        group = group.sort_values("score", ascending=False)

        if len(group) < 2 * k:
            continue

        long_leg = group.head(k)
        short_leg = group.tail(k)

        long_return = long_leg["next_return"].mean()
        short_return = short_leg["next_return"].mean()

        gross_return = long_return - short_return
        net_return = gross_return - 2 * transaction_cost

        daily_rows.append({
            "date": date,
            "long_return": long_return,
            "short_return": short_return,
            "gross_return": gross_return,
            "net_return": net_return,
            "n_available": len(group),
            "n_long": len(long_leg),
            "n_short": len(short_leg),
        })

    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)

    if daily.empty:
        raise ValueError("Backtest produced no daily returns.")

    daily["cum_gross_return"] = (1 + daily["gross_return"]).cumprod() - 1
    daily["cum_net_return"] = (1 + daily["net_return"]).cumprod() - 1

    return daily


def run_backtest_from_scores(
    meta_trade: pd.DataFrame,
    scores,
    returns: pd.DataFrame,
    k: int = 10,
    transaction_cost: float = 0.0005,
) -> pd.DataFrame:
    """
    Convenience function:
    use this right after model.predict_proba(X_trade).
    """

    predictions = build_predictions_df(meta_trade, scores)

    return run_backtest_from_predictions(
        predictions=predictions,
        returns=returns,
        k=k,
        transaction_cost=transaction_cost,
    )


def compute_metrics(daily_results: pd.DataFrame, return_col: str = "net_return") -> dict:
    """
    Compute performance metrics.
    """

    r = daily_results[return_col].dropna()

    if len(r) == 0:
        raise ValueError("No returns available.")

    avg_daily_return = r.mean()
    daily_volatility = r.std()

    sharpe_ratio = (
        np.sqrt(252) * avg_daily_return / daily_volatility
        if daily_volatility != 0
        else np.nan
    )

    cumulative_return = (1 + r).prod() - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(r)) - 1

    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    win_rate = (r > 0).mean()

    var_1 = r.quantile(0.01)
    cvar_1 = r[r <= var_1].mean()

    var_5 = r.quantile(0.05)
    cvar_5 = r[r <= var_5].mean()

    return {
        "avg_daily_return": avg_daily_return,
        "annualized_return": annualized_return,
        "daily_volatility": daily_volatility,
        "sharpe_ratio": sharpe_ratio,
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "VaR_1pct": var_1,
        "CVaR_1pct": cvar_1,
        "VaR_5pct": var_5,
        "CVaR_5pct": cvar_5,
        "num_days": len(r),
    }


def save_results(
    daily_results: pd.DataFrame,
    metrics: dict,
    model_name: str,
    output_dir: str = "results",
) -> None:
    """
    Save daily return series and summary metrics.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    daily_path = out / f"{model_name}_daily_returns.csv"
    metrics_path = out / f"{model_name}_metrics.csv"

    daily_results.to_csv(daily_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"Saved daily returns: {daily_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    print("Backtest module ready.")
    print("Use run_backtest_from_scores(meta_trade, scores, returns).")
