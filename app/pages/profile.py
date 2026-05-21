"""
profile.py 
Customer Profile page:
  • Metadata card (age, membership status)
  • LLM-generated taste profile card
  • Top-5 category bar chart
"""


from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.storage import load_table

PROCESSED_DIR = Path("data/processed")
DEMO_CACHE_DIR = Path("demo_cache")


def _load_taste_profile(customer_id: str, live: bool) -> str:
    """Load the cached or live-generated taste profile for a customer."""
    if live:
        from src.llm import generate_taste_profile
        return generate_taste_profile(customer_id)
    path = DEMO_CACHE_DIR / "taste_profiles.json"
    if path.exists():
        profiles = json.loads(path.read_text())
        return profiles.get(
            customer_id,
            "No cached taste profile for this customer.",
        )
    return "Cache not found."


def render():
    """Render the customer profile analysis page."""
    customer_id = st.session_state.get("customer_id", "")
    live_mode = st.session_state.get("live_mode", False)
    cold_start = st.session_state.get("cold_start", False)

    if cold_start:
        st.markdown(
            (
                "<div style='text-align: center; padding: 60px 20px;'>"
                "<h2 style='font-weight: 300;'>Welcome</h2>"
                "<p style='color: #666;'>Browsing in Guest Mode.</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    if not customer_id:
        return

    st.markdown(
        (
            "<h2 style='font-weight: 300; border-bottom: 1px solid #EAEAEA; "
            "padding-bottom: 15px; margin-bottom: 30px;'>PROFILE ANALYSIS</h2>"
        ),
        unsafe_allow_html=True,
    )

    # ── Metadata ────────────────────────────────────────────────────────
    try:
        customers = load_table(PROCESSED_DIR / "customers")
        row = customers[customers["customer_id"] == customer_id]
        if not row.empty:
            r = row.iloc[0]
            age = int(r.get("age", 0)) if pd.notna(r.get("age")) else "—"
            member = "Active" if r.get("club_member_status") == "ACTIVE" else "Inactive"

            st.markdown(
                f"""
                <div style="display: flex; gap: 40px; margin-bottom: 40px;">
                    <div><div style="font-size: 10px; color: #888; letter-spacing: 1px;">ID</div><div style="font-size: 14px; font-weight: bold;">{customer_id}</div></div>
                    <div><div style="font-size: 10px; color: #888; letter-spacing: 1px;">AGE</div><div style="font-size: 14px; font-weight: bold;">{age}</div></div>
                    <div><div style="font-size: 10px; color: #888; letter-spacing: 1px;">CLUB MEMBER</div><div style="font-size: 14px; font-weight: bold;">{member}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except (FileNotFoundError, KeyError, ValueError):
        pass

    # ── Style Archetypes & Taste Profile ────────────────────────────────
    st.markdown(
        (
            "<h4 style='font-size: 11px; letter-spacing: 1px; color: #888; "
            "margin-bottom: 10px;'>STYLE ARCHETYPE</h4>"
        ),
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 1, 3])
    for i, tag in enumerate(["SPORTY", "MINIMAL", "FUNCTIONAL"]):
        cols[i].markdown(
            (
                "<div style='border:1px solid #222; padding:4px 0; font-size:10px; "
                f"text-align:center; font-weight:bold;'>{tag}</div>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        (
            "<h4 style='font-size: 11px; letter-spacing: 1px; color: #888; "
            "margin-top:25px; margin-bottom: 15px;'>PERSONALIZED INSIGHTS</h4>"
        ),
        unsafe_allow_html=True,
    )

    profile_text = _load_taste_profile(customer_id, live_mode)
    st.markdown(f"""
    <div style="background:#F9F9F9; padding:20px; border-left:3px solid #E50010; font-size:14px; color:#222; line-height:1.6;">
        {profile_text}
    </div>
    """, unsafe_allow_html=True)

    # ── Category Chart ──────────────────────────────────────────────────
    st.markdown(
        (
            "<h4 style='font-size: 11px; letter-spacing: 1px; color: #888; "
            "margin-top:40px; margin-bottom: 20px;'>TOP CATEGORIES</h4>"
        ),
        unsafe_allow_html=True,
    )

    try:
        transactions = load_table(PROCESSED_DIR / "transactions")
        articles = load_table(PROCESSED_DIR / "articles").set_index("article_id")
        history = transactions[transactions["customer_id"] == customer_id]

        if not history.empty:
            cats = (
                history["article_id"]
                .map(articles["product_type_name"].to_dict())
                .value_counts()
                .head(6)
                .reset_index()
            )
            cats.columns = ["Category", "Count"]

            fig = px.bar(
                cats,
                x="Count",
                y="Category",
                orientation="h",
                color_discrete_sequence=['#E50010'],
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#000000', family='Noto Sans'),
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(
                    showgrid=False,
                    title=None,
                    categoryorder="total ascending",
                    tickfont=dict(color="#000000", size=12),
                    automargin=True,
                ),
                margin=dict(l=0, r=0, t=0, b=0), height=300
            )
            fig.update_traces(
                texttemplate='%{x}',
                textposition='outside',
                textfont_color='#000000',
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    except (FileNotFoundError, KeyError, ValueError):
        st.caption("No history data available.")
