"""
app.py — Streamlit entry point for the S&P 500 Statistical Arbitrage dashboard.

Handles:
    - Page config and global CSS
    - Sidebar navigation and model selection
    - Routing to page modules in src/app/

Run:
    streamlit run app.py
"""
import streamlit as st
from src.app.utils import get_available_models, load_config, MODEL_LABELS

# Page config
st.set_page_config(
    page_title="StatArb S&P 500",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stMetric {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1rem;
}
.stMetric label {
    color: #6c6c8a !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stMetric [data-testid="metric-container"] > div:nth-child(2) {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #e8e8f0;
}

div[data-testid="stSidebar"] {
    background: #0d0d18;
    border-right: 1px solid #1e1e2e;
}

.block-container { padding-top: 2rem; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4361ee;
    margin-bottom: 0.5rem;
}

.card {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

hr { border-color: #1e1e2e; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📈 StatArb")
    st.markdown(
        "<p style='color:#6c6c8a;font-size:0.8rem;margin-top:-0.5rem'>S&P 500 · Krauss et al. 2016</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "Performance", "Risk Diagnostics", "Simulation", "Sensitivity"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    available_models = get_available_models()
    selected_models = st.multiselect(
        "Models",
        options=available_models,
        default=available_models,
        format_func=lambda x: MODEL_LABELS.get(x, x),
    )

    cfg = load_config()
    st.markdown("---")
    st.markdown("<p class='section-header'>Config</p>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:#6c6c8a;font-size:0.8rem'>"
        f"k = {cfg['trading']['k']}<br>"
        f"TC = {cfg['trading']['transaction_cost_per_half_turn']*100:.3f}% per half-turn<br>"
        f"Batches = 1–22</p>",
        unsafe_allow_html=True,
    )

# Routing
if page == "Overview":
    from src.app.overview import render
    render()

elif page == "Performance":
    from src.app.performance import render
    render(selected_models)

elif page == "Risk Diagnostics":
    from src.app.risk import render
    render(selected_models)

elif page == "Simulation":
    from src.app.simulation import render
    render()

elif page == "Sensitivity":
    from src.app.sensitivity import render
    render()