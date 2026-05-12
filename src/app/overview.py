"""
src/app/overview.py — Overview page.
"""

import pandas as pd
import streamlit as st

from src.app.utils import (
    PAPER_RESULTS,
    MODEL_LABELS,
    get_available_models,
    load_metrics,
)


def render():
    st.markdown("# Statistical Arbitrage on the S&P 500")
    st.markdown("<p style='color:#6c6c8a'>Reproducing & extending Krauss, Do & Huck (2016) with LSTM and CNN architectures</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### The Original Paper")
        st.markdown("""
        <div class='card'>
        Krauss et al. (2016) train DNN, GBT, and Random Forest models on <b>31 hand-crafted lagged
        return features</b> to forecast which S&P 500 stocks will outperform the cross-sectional median
        the next day. Using a sliding window of 750-day training / 250-day test periods (23 batches,
        1992–2015), they go <b>long top-k</b> and <b>short bottom-k</b> stocks ranked by predicted probability.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Our Extensions")
        st.markdown("""
        <div class='card'>
        We introduce two new architectures that operate on <b>raw return sequences</b> rather than
        pre-engineered features:<br><br>
        <b>LSTM</b> — Tests whether long-range sequential memory captures dependencies that
        hand-crafted lag aggregation misses.<br><br>
        <b>1D CNN</b> — Motivated by the paper's variable importance finding that R(1)–R(5) dominates.
        Convolutional filters act as learned adaptive sliding windows over the return sequence.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Paper Benchmark (post-cost, k=10)")
        paper_df = pd.DataFrame(PAPER_RESULTS).T.reset_index()
        paper_df.columns = ["Model", "Avg Daily Ret", "Sharpe", "Max DD", "Win Rate"]
        paper_df["Avg Daily Ret"] = paper_df["Avg Daily Ret"].map(lambda x: f"{x*100:.4f}%")
        paper_df["Sharpe"] = paper_df["Sharpe"].map(lambda x: f"{x:.2f}")
        paper_df["Max DD"] = paper_df["Max DD"].map(lambda x: f"{x*100:.1f}%")
        paper_df["Win Rate"] = paper_df["Win Rate"].map(lambda x: f"{x*100:.1f}%")
        st.dataframe(paper_df.set_index("Model"), use_container_width=True)

        st.markdown("### Our Results (post-cost, k=10)")
        our_rows = []
        for m in get_available_models():
            metrics = load_metrics(m)
            if metrics:
                our_rows.append({
                    "Model":        MODEL_LABELS.get(m, m),
                    "Avg Daily Ret": f"{metrics['avg_daily_return']*100:.4f}%",
                    "Sharpe":       f"{metrics['sharpe_ratio']:.2f}",
                    "Max DD":       f"{metrics['max_drawdown']*100:.1f}%",
                    "Win Rate":     f"{metrics['win_rate']*100:.1f}%",
                })
        if our_rows:
            st.dataframe(pd.DataFrame(our_rows).set_index("Model"), use_container_width=True)