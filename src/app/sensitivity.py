"""Sensitivity Analysis page."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.app.utils import (
    COLORS,
    MODEL_LABELS,
    PLOT_LAYOUT,
    get_available_models,
    load_all_predictions,
    load_config,
    load_returns,
)
from src.backtest.backtest import run_backtest_from_predictions, compute_metrics


def render():
    st.markdown("# Sensitivity Analysis")
    st.markdown("---")

    available_models = get_available_models()
    cfg = load_config()

    col1, col2 = st.columns(2)
    with col1:
        sens_model = st.selectbox(
            "Model",
            options=available_models,
            format_func=lambda x: MODEL_LABELS.get(x, x),
        )
    with col2:
        sens_type = st.radio(
            "Vary",
            ["Portfolio size k", "Transaction cost"],
            horizontal=True,
        )

    if st.button("▶  Run Sensitivity", type="primary"):
        returns = load_returns()
        if returns is None:
            st.error("returns_clean.parquet not found.")
            return

        predictions = load_all_predictions(sens_model)
        if predictions is None:
            st.error("No predictions found.")
            return

        results_rows = []
        color = COLORS.get(sens_model, "#4361ee")

        if sens_type == "Portfolio size k":
            k_values  = [1, 5, 10, 25, 50, 75, 100]
            fixed_tc  = cfg["trading"]["transaction_cost_per_half_turn"]
            progress  = st.progress(0)

            for idx, k_val in enumerate(k_values):
                try:
                    daily = run_backtest_from_predictions(
                        predictions=predictions, returns=returns,
                        k=k_val, transaction_cost=fixed_tc,
                    )
                    m = compute_metrics(daily, return_col="net_return")
                    results_rows.append({
                        "k":               k_val,
                        "Sharpe":          m["sharpe_ratio"],
                        "Avg Daily Ret (%)": m["avg_daily_return"] * 100,
                        "Max Drawdown (%)": m["max_drawdown"] * 100,
                        "Win Rate (%)":    m["win_rate"] * 100,
                    })
                except Exception:
                    pass
                progress.progress((idx + 1) / len(k_values))

            if results_rows:
                sens_df = pd.DataFrame(results_rows)
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=["Sharpe Ratio", "Avg Daily Return (%)",
                                    "Max Drawdown (%)", "Win Rate (%)"],
                )
                for col_name, r, c in [
                    ("Sharpe", 1, 1), ("Avg Daily Ret (%)", 1, 2),
                    ("Max Drawdown (%)", 2, 1), ("Win Rate (%)", 2, 2),
                ]:
                    fig.add_trace(go.Scatter(
                        x=sens_df["k"], y=sens_df[col_name],
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=8),
                        showlegend=False,
                    ), row=r, col=c)
                fig.update_layout(
                    **PLOT_LAYOUT, height=500,
                    title=f"{MODEL_LABELS.get(sens_model, sens_model)} — Sensitivity to k",
                )
                fig.update_xaxes(title_text="k")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(sens_df.set_index("k"), use_container_width=True)

        else:  # Transaction cost
            tc_values = [0, 1, 2, 5, 10, 20, 30, 50]
            fixed_k   = cfg["trading"]["k"]
            progress  = st.progress(0)

            for idx, tc_bps in enumerate(tc_values):
                tc = tc_bps / 10000
                try:
                    daily = run_backtest_from_predictions(
                        predictions=predictions, returns=returns,
                        k=fixed_k, transaction_cost=tc,
                    )
                    m = compute_metrics(daily, return_col="net_return")
                    results_rows.append({
                        "TC (bps)":        tc_bps,
                        "Sharpe":          m["sharpe_ratio"],
                        "Avg Daily Ret (%)": m["avg_daily_return"] * 100,
                        "Max Drawdown (%)": m["max_drawdown"] * 100,
                        "Win Rate (%)":    m["win_rate"] * 100,
                    })
                except Exception:
                    pass
                progress.progress((idx + 1) / len(tc_values))

            if results_rows:
                sens_df = pd.DataFrame(results_rows)
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=["Sharpe Ratio", "Avg Daily Return (%)",
                                    "Max Drawdown (%)", "Win Rate (%)"],
                )
                for col_name, r, c in [
                    ("Sharpe", 1, 1), ("Avg Daily Ret (%)", 1, 2),
                    ("Max Drawdown (%)", 2, 1), ("Win Rate (%)", 2, 2),
                ]:
                    fig.add_trace(go.Scatter(
                        x=sens_df["TC (bps)"], y=sens_df[col_name],
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=8),
                        showlegend=False,
                    ), row=r, col=c)
                fig.update_layout(
                    **PLOT_LAYOUT, height=500,
                    title=f"{MODEL_LABELS.get(sens_model, sens_model)} — Sensitivity to Transaction Cost",
                )
                fig.update_xaxes(title_text="Transaction Cost (bps)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(sens_df.set_index("TC (bps)"), use_container_width=True)

    # Planned extensions note for sequential models
    if sens_model in ("lstm", "cnn"):
        st.markdown("---")
        st.markdown("### Planned Sensitivity Extensions for Sequential Models")
        st.info("""
        The following sensitivity analyses are planned for LSTM and CNN but require retraining:

        - **Sequence Length** — 60 / 120 / 240 days. The README flags that 240-step sequences
          risk vanishing gradients in the LSTM. Shorter sequences (60–120) are hypothesised
          to improve performance.

        - **Kernel Size (CNN)** — 3 / 5 / 10 / 20. Controls the local pattern detection window.
          Smaller kernels (3, 5) align with the paper's R(1)–R(5) variable importance finding.

        - **Architecture depth** — 1 / 2 / 3 layers for both LSTM and CNN.

        - **With vs without Optuna tuning** — The original paper used fixed DNN hyperparameters
          without tuning. Our Optuna integration directly addresses this weakness.
        """)