"""Performance page."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.app.utils import (
    COLORS,
    MODEL_LABELS,
    PLOT_LAYOUT,
    load_daily_results,
    load_metrics,
)


def render(selected_models: list[str]):
    st.markdown("# Performance")
    st.markdown("---")

    if not selected_models:
        st.warning("Select at least one model from the sidebar.")
        return

    # Metrics
    cols = st.columns(len(selected_models))
    for col, model in zip(cols, selected_models):
        metrics = load_metrics(model)
        if metrics:
            with col:
                st.metric(
                    MODEL_LABELS.get(model, model),
                    f"{metrics['sharpe_ratio']:.2f}",
                    f"{metrics['avg_daily_return']*100:.3f}%/day",
                    help="Annualised Sharpe / Avg daily net return",
                )

    st.markdown("---")

    # Equity curves
    st.markdown("### Cumulative Net Returns")
    fig = go.Figure()
    for model in selected_models:
        daily = load_daily_results(model)
        if daily is not None:
            fig.add_trace(go.Scatter(
                x=daily["date"],
                y=daily["cum_net_return"] * 100,
                name=MODEL_LABELS.get(model, model),
                line=dict(color=COLORS.get(model, "#ffffff"), width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>"
                              + MODEL_LABELS.get(model, model) + "</extra>",
            ))
    fig.update_layout(**PLOT_LAYOUT, title="Cumulative Net Return (%)", height=420)
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # Daily return distribution
    with col1:
        st.markdown("### Daily Return Distribution")
        fig2 = go.Figure()
        for model in selected_models:
            daily = load_daily_results(model)
            if daily is not None:
                fig2.add_trace(go.Histogram(
                    x=daily["net_return"] * 100,
                    name=MODEL_LABELS.get(model, model),
                    opacity=0.6,
                    nbinsx=80,
                    marker_color=COLORS.get(model, "#ffffff"),
                ))
        fig2.update_layout(**PLOT_LAYOUT, barmode="overlay", height=340,
                           xaxis_title="Daily Net Return (%)")
        st.plotly_chart(fig2, use_container_width=True)

    # Rolling Sharpe
    with col2:
        st.markdown("### 252-Day Rolling Sharpe")
        fig3 = go.Figure()
        for model in selected_models:
            daily = load_daily_results(model)
            if daily is not None:
                r = daily.set_index("date")["net_return"]
                rolling_sharpe = (
                    r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
                )
                fig3.add_trace(go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    name=MODEL_LABELS.get(model, model),
                    line=dict(color=COLORS.get(model, "#ffffff"), width=1.5),
                ))
        fig3.add_hline(y=0, line_dash="dot", line_color="#6c6c8a")
        fig3.update_layout(**PLOT_LAYOUT, height=340, yaxis_title="Sharpe Ratio")
        st.plotly_chart(fig3, use_container_width=True)

    # Full metrics table
    st.markdown("### Full Metrics Table")
    metric_rows = []
    for model in selected_models:
        metrics = load_metrics(model)
        if metrics:
            metric_rows.append({
                "Model":        MODEL_LABELS.get(model, model),
                "Avg Daily Ret": f"{metrics['avg_daily_return']*100:.4f}%",
                "Ann. Return":   f"{metrics['annualized_return']*100:.1f}%",
                "Sharpe":        f"{metrics['sharpe_ratio']:.4f}",
                "Volatility":    f"{metrics['daily_volatility']*100:.4f}%",
                "Max Drawdown":  f"{metrics['max_drawdown']*100:.1f}%",
                "Win Rate":      f"{metrics['win_rate']*100:.1f}%",
                "VaR 1%":        f"{metrics['VaR_1pct']*100:.2f}%",
                "CVaR 1%":       f"{metrics['CVaR_1pct']*100:.2f}%",
            })
    if metric_rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(metric_rows).set_index("Model"), use_container_width=True)