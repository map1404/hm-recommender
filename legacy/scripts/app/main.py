"""
main.py
-------
Frontend Developer (Lei) module.

Streamlit multi-page app entry point.
Pages:
  - Profile         : customer metadata + LLM taste profile + charts
  - Recommendations : top-10 item grid with real images + explanation badges
  - Explore Styles  : style archetypes browser
"""

import sys
import json
import argparse
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# CLI flags (pass after `--` in `streamlit run app/main.py -- --live`)
parser = argparse.ArgumentParser()
parser.add_argument("--live", action="store_true", default=False)
args, _ = parser.parse_known_args()

DEMO_CACHE_DIR = ROOT_DIR / "demo_cache"
TASTE_PROFILES_PATH = DEMO_CACHE_DIR / "taste_profiles.json"

# Load curated customer IDs from cache
if TASTE_PROFILES_PATH.exists():
    _profiles = json.loads(TASTE_PROFILES_PATH.read_text())
    DEMO_CUSTOMER_IDS = list(_profiles.keys())
else:
    DEMO_CUSTOMER_IDS = []

st.set_page_config(
    page_title="H&M Style Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: #e8e8f0 !important; }
    .stApp { background-color: #f8f7f4; }
    .brand-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #e63946 100%);
        color: white !important;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .brand-header h1 { color: white !important; margin: 0; font-size: 2rem; }
    .brand-header p { color: rgba(255,255,255,0.85) !important; margin: 0.3rem 0 0; }
    .profile-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    .taste-profile-box {
        background: linear-gradient(135deg, #fff9f0, #fff3e0);
        border-left: 4px solid #e63946;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.25rem;
        font-size: 1rem;
        color: #333;
        line-height: 1.65;
        margin: 1rem 0;
    }
    .explanation-badge {
        background: linear-gradient(135deg, #EEF4FF, #e8f0fe);
        border-left: 3px solid #4A7CF7;
        border-radius: 0 6px 6px 0;
        padding: 8px 10px;
        font-size: 0.82rem;
        color: #1a1a2e;
        margin-top: 0.6rem;
        line-height: 1.4;
    }
    .score-pill {
        background: #e9ecef;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #555;
        display: inline-block;
        margin-top: 4px;
    }
    .cat-tag {
        background: #f1f3f5;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.72rem;
        color: #666;
        margin: 2px 2px 0 0;
        display: inline-block;
    }
    hr { border: none; border-top: 1px solid #e8e8e8; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/53/H%26M-Logo.svg",
        width=80,
    )
    st.title("H&M Recommender")
    st.caption("Personalised taste profiles powered by hybrid AI")

    if DEMO_CUSTOMER_IDS:
        customer_id = st.selectbox(
            "Select a customer",
            DEMO_CUSTOMER_IDS,
            format_func=lambda x: x[:24] + "..." if len(x) > 24 else x,
        )
    else:
        customer_id = st.text_input(
            "Enter Customer ID",
            placeholder="Paste a customer_id from the dataset",
        )
        if not customer_id:
            st.warning("No demo cache found. Run `python src/inference.py --cache` first.")

    live_mode = args.live or st.toggle(
        "⚡ Live LLM mode",
        value=False,
        help="Disables caching — adds ~30s latency",
    )

    st.session_state["customer_id"] = customer_id
    st.session_state["live_mode"] = live_mode

    st.divider()
    page = st.radio(
        "Navigate",
        ["👤 Profile", "✨ Recommendations", "🎨 Explore Styles"],
        label_visibility="collapsed",
    )
    st.session_state["current_page"] = page

    st.markdown(
        "<div style='font-size:0.75rem;color:#aaa;padding-top:1rem;'>"
        "H&M Recommender · Recommender Systems Final Project</div>",
        unsafe_allow_html=True,
    )

# Page routing
if page == "👤 Profile":
    from pages.profile import render
    render()
elif page == "✨ Recommendations":
    from pages.recommendations import render
    render()
elif page == "🎨 Explore Styles":
    from pages.explore import render
    render()
