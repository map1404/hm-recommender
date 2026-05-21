"""
profile.py
----------
Frontend Developer (Lei) module.

Customer Profile page:
  - Metadata card (age, membership status)
  - LLM-generated taste profile
  - Top categories bar chart
  - Colour palette pie chart
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from src.storage import load_table

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data/processed"
DEMO_CACHE_DIR = ROOT_DIR / "demo_cache"


@st.cache_data
def _load_taste_profiles():
    path = DEMO_CACHE_DIR / "taste_profiles.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def render():
    customer_id = st.session_state.get("customer_id", "")
    live_mode = st.session_state.get("live_mode", False)

    st.markdown("""
    <div class="brand-header">
        <h1>👤 Customer Profile</h1>
        <p>Personalised style insights powered by purchase history & AI</p>
    </div>
    """, unsafe_allow_html=True)

    if not customer_id:
        st.info("Select a customer from the sidebar to begin.")
        return

    # Metadata card
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown("#### 🪪 Customer")
        st.markdown(f"**ID:** `{customer_id[:20]}…`")
        try:
            customers = load_table(PROCESSED_DIR / "customers")
            row = customers[customers["customer_id"] == customer_id]
            if not row.empty:
                r = row.iloc[0]
                age = r.get("age")
                if age and not pd.isna(age):
                    st.markdown(f"**Age:** {int(age)}")
                club = r.get("club_member_status", "")
                if club:
                    st.markdown(f"**Club Status:** {club}")
        except Exception:
            pass
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### ✨ Taste Profile")

        if live_mode:
            with st.spinner("Generating taste profile with AI..."):
                try:
                    from src.llm import generate_taste_profile
                    profile = generate_taste_profile(customer_id)
                except Exception as e:
                    st.error(f"Error: {e}")
                    profile = ""
        else:
            profiles = _load_taste_profiles()
            profile = profiles.get(customer_id, "")

        if profile:
            st.markdown(
                f'<div class="taste-profile-box">{profile}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No taste profile cached. Enable Live LLM mode or run `python src/inference.py --cache`.")

    # Charts from purchase history
    st.markdown("---")
    st.subheader("📊 Style Insights")

    try:
        transactions = load_table(PROCESSED_DIR / "transactions")
        articles = load_table(PROCESSED_DIR / "articles").set_index("article_id")
        history = transactions[transactions["customer_id"] == customer_id]

        if not history.empty:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Top Categories**")
                cats = (
                    history["article_id"]
                    .map(articles["product_type_name"].to_dict())
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                cats.columns = ["Category", "Count"]
                fig = px.bar(
                    cats, x="Count", y="Category", orientation="h",
                    color="Count", color_continuous_scale=["#f0e6ff", "#e63946"],
                    template="plotly_white",
                )
                fig.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    yaxis={"categoryorder": "total ascending"},
                    margin=dict(l=0, r=0, t=10, b=10), height=280,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("**Colour Palette**")
                colours = (
                    history["article_id"]
                    .map(articles["colour_group_name"].to_dict())
                    .value_counts()
                    .head(8)
                    .reset_index()
                )
                colours.columns = ["Colour", "Count"]
                fig2 = px.pie(
                    colours, values="Count", names="Colour", hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig2.update_layout(
                    margin=dict(l=0, r=0, t=10, b=10), height=280,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("No purchase history found for this customer.")
    except FileNotFoundError:
        st.caption("Processed data not found. Run `python src/preprocessing.py` first.")
