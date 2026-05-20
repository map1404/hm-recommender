"""
main.py — Lei
Streamlit app entry point.

Routing:
  / → Customer Profile
  /recommendations → Recommendations Grid
"""

import sys
import json
import argparse
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.storage import load_table

# ── CLI flags (pass after `--` in `streamlit run app/main.py -- --live`)
parser = argparse.ArgumentParser()
parser.add_argument("--live", action="store_true", default=False)
args, _ = parser.parse_known_args()

DEMO_CACHE_DIR = Path("demo_cache")
TASTE_PROFILES_PATH = DEMO_CACHE_DIR / "taste_profiles.json"
PROCESSED_DIR = Path("data/processed")

# Load curated customer IDs from cache
if TASTE_PROFILES_PATH.exists():
    _profiles = json.loads(TASTE_PROFILES_PATH.read_text())
    DEMO_CUSTOMER_IDS = list(_profiles.keys())
else:
    DEMO_CUSTOMER_IDS = []

try:
    CUSTOMER_INDEX = load_table(PROCESSED_DIR / "customer_index")
    ALL_CUSTOMER_IDS = CUSTOMER_INDEX["customer_id"].tolist()
except FileNotFoundError:
    ALL_CUSTOMER_IDS = []

st.set_page_config(
    page_title="H&M Taste Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/53/H%26M-Logo.svg", width=80)
    st.title("H&M Recommender")
    st.caption("Personalized taste profiles powered by hybrid AI")

    customer_id = st.text_input(
        "Enter customer ID",
        value=st.session_state.get("customer_id", ""),
        placeholder="Paste a customer_id from the dataset",
    ).strip()

    if DEMO_CUSTOMER_IDS:
        demo_customer_id = st.selectbox(
            "Or choose a demo customer",
            [""] + DEMO_CUSTOMER_IDS,
            format_func=lambda x: "Choose a demo user..." if not x else x[:20] + "...",
        )
        if demo_customer_id:
            customer_id = demo_customer_id
        st.caption("Cached profiles/explanations are available for demo users.")

    if ALL_CUSTOMER_IDS:
        helper_customer_id = st.selectbox(
            "Or pick a known customer ID",
            [""] + ALL_CUSTOMER_IDS,
            format_func=lambda x: "Choose from dataset..." if not x else x[:20] + "...",
        )
        if helper_customer_id:
            customer_id = helper_customer_id
        st.caption("Use live mode for non-demo users if no cached profile is available.")

    live_mode = args.live or st.toggle("Live LLM mode", value=False,
                                        help="Disables caching — adds ~30s latency")

    st.session_state["customer_id"] = customer_id
    st.session_state["live_mode"] = live_mode

    st.divider()
    page = st.radio("Navigate", ["Profile", "Recommendations"], label_visibility="collapsed")

# ── Page routing ───────────────────────────────────────────────────────────────
if page == "Profile":
    from app.pages.profile import render
    render()
else:
    from app.pages.recommendations import render
    render()
