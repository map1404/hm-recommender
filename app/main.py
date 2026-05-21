"""
main.py 
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

import __main__
from src.demo_artifacts import DemoALS

# Trick pickle into finding DemoALS in the current main script
__main__.DemoALS = DemoALS

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.storage import load_table

# ── CLI flags
parser = argparse.ArgumentParser()
parser.add_argument("--live", action="store_true", default=False)
args, _ = parser.parse_known_args()

DEMO_CACHE_DIR = Path("demo_cache")
TASTE_PROFILES_PATH = DEMO_CACHE_DIR / "taste_profiles.json"
PROCESSED_DIR = Path("data/processed")

# Load curated customer IDs
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

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="H&M Personalized Stylist",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

# ── Custom H&M Official Website CSS ─────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Hide Streamlit default UI */
        [data-testid="stSidebarNav"], [data-testid="collapsedControl"] {display: none !important;}
        header {display: none !important;}
        footer {display: none !important;}
        
        /* H&M Brand Colors & Typography */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@300;400;700&display=swap');
        
        :root {
            --hm-red: #E50010;
            --hm-text: #222222;
            --hm-bg: #FFFFFF;
            --hm-gray: #F4F4F4;
        }
        
        .stApp {
            background-color: var(--hm-bg);
            color: var(--hm-text);
            font-family: 'Noto Sans', Arial, sans-serif;
            padding-top: 0px !important;
        }

        .block-container {
            padding-top: 0px !important;
        }

        /* Top Promo Banner */
        .promo-banner {
            background-color: var(--hm-gray);
            color: var(--hm-text);
            text-align: center;
            font-size: 11px;
            padding: 8px 0;
            letter-spacing: 0.5px;
            width: 100%;
        }

        /* Header Layout */
        .hm-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px 0 20px 0;
        }

        /* Fake Navigation Menu */
        .hm-nav {
            display: flex;
            gap: 25px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1px;
            margin-top: 20px;
            margin-bottom: 20px;
            border-bottom: 1px solid #EAEAEA;
            padding-bottom: 15px;
            width: 100%;
            justify-content: center;
        }
        .hm-nav span {
            cursor: pointer;
            color: var(--hm-text);
        }
        .hm-nav span.active-tab {
            color: var(--hm-red);
            border-bottom: 2px solid var(--hm-red);
            padding-bottom: 13px;
        }

        /* Streamlit Input Overrides */
        div[data-baseweb="select"] > div {
            border-radius: 0px !important;
            border: 1px solid var(--hm-text) !important;
        }
        input {
            border-radius: 0px !important;
            border: 1px solid var(--hm-text) !important;
        }
        /* Style Streamlit Tabs to look minimalist */
        button[data-baseweb="tab"] {
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 1px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── H&M Official Header Components ──────────────────────────────────────────
# 1. Promo Banner
st.markdown('<div class="promo-banner">Free shipping over $40 & Free returns for members</div>', unsafe_allow_html=True)

# 2. Main Logo & Faux Navigation
st.markdown(
    """
    <div class="hm-header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/H%26M-Logo.svg" width="65">
        <div class="hm-nav">
            <span>WOMEN</span>
            <span>MEN</span>
            <span>BABY</span>
            <span>KIDS</span>
            <span>H&M HOME</span>
            <span>SPORT</span>
            <span>SALE</span>
            <span class="active-tab">AI STYLIST</span>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# ── Customer Login / Selection Area ─────────────────────────────────────────
# We use columns to center the styling tools exactly like a web container
spacer_left, main_content, spacer_right = st.columns([1, 2, 1])

with main_content:
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px; font-weight: 300;'>Your Personal AI Stylist</h3>", unsafe_allow_html=True)
    
    with st.expander("MY ACCOUNT / SETTINGS", expanded=True):
        user_mode = st.radio(
            "Account Status",
            ["Existing customer", "Guest (Cold-start user)"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
        cold_start = user_mode.startswith("Guest")

        if cold_start:
            customer_id = "cold_start"
            st.caption("Browsing as a guest. We are showing you the most popular items globally.")
        else:
            input_options = ["Enter Customer ID"]
            if DEMO_CUSTOMER_IDS:
                input_options.insert(0, "Choose Demo Account") 
            if ALL_CUSTOMER_IDS:
                input_options.append("Randomly Select Known Customer")
            
            input_method = st.radio("Authentication Method:", input_options, horizontal=True)
            
            customer_id = ""
            
            if input_method == "Choose Demo Account":
                customer_id = st.selectbox(
                    "Select a curated profile:",
                    [""] + DEMO_CUSTOMER_IDS,
                    format_func=lambda x: "Choose a demo profile..." if not x else x[:20] + "...",
                    label_visibility="collapsed"
                )
                
            elif input_method == "Enter Customer ID":
                default_val = st.session_state.get("customer_id", "")
                if default_val == "cold_start":
                    default_val = ""
                    
                customer_id = st.text_input(
                    "Customer ID:",
                    value=default_val,
                    placeholder="Enter your exact Customer ID",
                    label_visibility="collapsed"
                ).strip()
                
            elif input_method == "Randomly Select Known Customer":
                customer_id = st.selectbox(
                    "Select from database:",
                    [""] + ALL_CUSTOMER_IDS,
                    format_func=lambda x: "Randomize from database..." if not x else x[:20] + "...",
                    label_visibility="collapsed"
                )

        st.divider()
        col_nav1, col_nav2 = st.columns([2, 1])
        with col_nav1:
            page = st.radio("Navigate:", [ "Profile Analysis","Recommendations"], horizontal=True, label_visibility="collapsed")
        with col_nav2:
            live_mode = args.live or st.toggle("Enable Live AI Generation", value=False)

        st.session_state["customer_id"] = customer_id
        st.session_state["live_mode"] = live_mode
        st.session_state["cold_start"] = cold_start

st.write("<br><br>", unsafe_allow_html=True) # Deep Spacer before content

# ── Page routing ───────────────────────────────────────────────────────────────
if page == "Profile Analysis":
    from app.pages.profile import render
    render()
else:
    from app.pages.recommendations import render
    render()