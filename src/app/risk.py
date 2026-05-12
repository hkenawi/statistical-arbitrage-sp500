"""Risk Diagnostics page."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.utils import (
    COLORS,
    MODEL_LABELS,
    PAPER_RESULTS,
    PLOT_LAYOUT,
    load_daily_results,
    load_metrics,
)


def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render(selected_models: list[str]):
    st.markdown("# Risk Diagnostics")
    st.markdown("---")

    if not selected_models:
        st.warning("Select at least one model from the sidebar.")
        return

    col1, col2 = st.columns(2)

    # Drawdown
    with col1:
        st.markdown("### Drawdown")
        fig = go.Figure()
        for model in selected_models:
            daily = load_daily_results(model)
            if daily is not None:
                equity   = (1 + daily["net_return"]).cumprod()
                drawdown = (equity / equity.cummax() - 1) * 100
                color    = COLORS.get(model, "#ffffff")
                fig.add_trace(go.Scatter(
                    x=daily["date"],
                    y=drawdown,
                    name=MODEL_LABELS.get(model, model),
                    line=dict(color=color, width=1.5),
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(color, 0.1),
                ))
        fig.update_layout(**PLOT_LAYOUT, height=340, yaxis_title="Drawdown (%)")
        st.plotly_chart(fig, use_container_width=True)

    # VaR / CVaR
    with col2:
        st.markdown("### VaR & CVaR (1%)")
        var_data = []
        for model in selected_models:
            metrics = load_metrics(model)
            if metrics:
                var_data.append({
                    "Model":   MODEL_LABELS.get(model, model),
                    "VaR 1%":  abs(metrics["VaR_1pct"]) * 100,
                    "CVaR 1%": abs(metrics["CVaR_1pct"]) * 100,
                })
        if var_data:
            var_df = pd.DataFrame(var_data)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=var_df["Model"], y=var_df["VaR 1%"],
                name="VaR 1%", marker_color="#4361ee",
            ))
            fig2.add_trace(go.Bar(
                x=var_df["Model"], y=var_df["CVaR 1%"],
                name="CVaR 1%", marker_color="#f72585",
            ))
            fig2.update_layout(**PLOT_LAYOUT, barmode="group", height=340,
                               yaxis_title="Loss (%)")
            st.plotly_chart(fig2, use_container_width=True)

    # Long vs Short contribution
    st.markdown("### Long vs Short Leg Contribution")
    fig3 = go.Figure()
    for model in selected_models:
        daily = load_daily_results(model)
        if daily is not None and "long_return" in daily.columns:
            color     = COLORS.get(model, "#ffffff")
            long_cum  = (1 + daily["long_return"]).cumprod() - 1
            short_cum = (1 + daily["short_return"].abs()).cumprod() - 1
            fig3.add_trace(go.Scatter(
                x=daily["date"], y=long_cum * 100,
                name=f"{MODEL_LABELS.get(model, model)} Long",
                line=dict(color=color, width=1.2),
            ))
            fig3.add_trace(go.Scatter(
                x=daily["date"], y=short_cum * 100,
                name=f"{MODEL_LABELS.get(model, model)} Short",
                line=dict(color=color, width=1.2, dash="dot"),
            ))
    fig3.update_layout(**PLOT_LAYOUT, height=360, yaxis_title="Cumulative Return (%)")
    st.plotly_chart(fig3, use_container_width=True)

    # Risk metrics table
    st.markdown("### Risk Metrics vs Paper Benchmark")
    rows = []
    for model in selected_models:
        metrics = load_metrics(model)
        if metrics:
            label = MODEL_LABELS.get(model, model)
            paper = PAPER_RESULTS.get(label, {})
            rows.append({
                "Model":          label,
                "Max DD (Ours)":  f"{metrics['max_drawdown']*100:.1f}%",
                "Max DD (Paper)": f"{paper.get('max_drawdown', float('nan'))*100:.1f}%" if paper else "—",
                "VaR 1%":         f"{metrics['VaR_1pct']*100:.2f}%",
                "CVaR 1%":        f"{metrics['CVaR_1pct']*100:.2f}%",
                "VaR 5%":         f"{metrics['VaR_5pct']*100:.2f}%",
                "CVaR 5%":        f"{metrics['CVaR_5pct']*100:.2f}%",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)