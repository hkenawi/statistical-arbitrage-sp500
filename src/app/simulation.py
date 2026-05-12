"""Run New Simulation page."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.utils import (
    COLORS,
    MODEL_LABELS,
    PLOT_LAYOUT,
    get_available_models,
    load_all_predictions,
    load_returns,
)
from src.backtest.backtest import run_backtest_from_predictions, compute_metrics


def render():
    st.markdown("# Run New Simulation")
    st.markdown(
        "<p style='color:#6c6c8a'>Re-run the backtest with custom parameters using saved model predictions</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    available_models = get_available_models()

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_k = st.slider(
            "Portfolio size k",
            min_value=1, max_value=100, value=10,
            help="Number of long and short positions",
        )
    with col2:
        sim_tc = st.slider(
            "Transaction cost (bps per half-turn)",
            min_value=0, max_value=50, value=5,
            help="Cost in basis points per half-turn",
        ) / 10000
    with col3:
        sim_models = st.multiselect(
            "Models to simulate",
            options=available_models,
            default=available_models[:3] if len(available_models) >= 3 else available_models,
            format_func=lambda x: MODEL_LABELS.get(x, x),
        )

    if st.button("▶  Run Simulation", type="primary"):
        returns = load_returns()
        if returns is None:
            st.error("returns_clean.parquet not found.")
            return

        sim_results = {}
        progress = st.progress(0)

        for idx, model in enumerate(sim_models):
            predictions = load_all_predictions(model)
            if predictions is None:
                st.warning(f"No predictions found for {MODEL_LABELS.get(model, model)}")
                continue
            try:
                daily = run_backtest_from_predictions(
                    predictions=predictions,
                    returns=returns,
                    k=sim_k,
                    transaction_cost=sim_tc,
                )
                sim_results[model] = daily
            except Exception as e:
                st.error(f"{MODEL_LABELS.get(model, model)}: {e}")
            progress.progress((idx + 1) / len(sim_models))

        if not sim_results:
            st.warning("No results — check prediction files.")
            return

        st.markdown("### Simulation Results")

        # Equity curves
        fig = go.Figure()
        for model, daily in sim_results.items():
            fig.add_trace(go.Scatter(
                x=daily["date"],
                y=daily["cum_net_return"] * 100,
                name=MODEL_LABELS.get(model, model),
                line=dict(color=COLORS.get(model, "#ffffff"), width=1.5),
            ))
        fig.update_layout(
            **PLOT_LAYOUT, height=400,
            title=f"Equity Curves — k={sim_k}, TC={sim_tc*10000:.0f}bps",
            yaxis_title="Cumulative Net Return (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        sim_metric_rows = []
        for model, daily in sim_results.items():
            m = compute_metrics(daily, return_col="net_return")
            sim_metric_rows.append({
                "Model":         MODEL_LABELS.get(model, model),
                "Avg Daily Ret": f"{m['avg_daily_return']*100:.4f}%",
                "Ann. Return":   f"{m['annualized_return']*100:.1f}%",
                "Sharpe":        f"{m['sharpe_ratio']:.4f}",
                "Max Drawdown":  f"{m['max_drawdown']*100:.1f}%",
                "Win Rate":      f"{m['win_rate']*100:.1f}%",
            })
        st.dataframe(
            pd.DataFrame(sim_metric_rows).set_index("Model"),
            use_container_width=True,
        )