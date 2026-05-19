"""
profile.py — Lei
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
    if live:
        from src.llm import generate_taste_profile
        return generate_taste_profile(customer_id)

    path = DEMO_CACHE_DIR / "taste_profiles.json"
    if path.exists():
        profiles = json.loads(path.read_text())
        return profiles.get(
            customer_id,
            "No cached taste profile for this customer. Demo cache only covers the curated demo users; enable live mode for on-demand generation.",
        )
    return "Cache not found. Run `python src/inference.py --cache` first."


def render():
    customer_id = st.session_state.get("customer_id", "")
    live_mode = st.session_state.get("live_mode", False)

    if not customer_id:
        st.info("Select a customer from the sidebar to begin.")
        return

    st.header("Customer Profile")

    # ── Metadata card ──────────────────────────────────────────────────────────
    try:
        customers = load_table(PROCESSED_DIR / "customers")
        row = customers[customers["customer_id"] == customer_id]
        if not row.empty:
            r = row.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Customer ID", customer_id[:16] + "…")
            col2.metric("Age", int(r.get("age", 0)) if pd.notna(r.get("age")) else "—")
            col3.metric("Club member", "Yes" if r.get("club_member_status") == "ACTIVE" else "No")
    except FileNotFoundError:
        st.caption("customers.parquet not found — run preprocessing first.")

    st.divider()

    # ── Taste profile ──────────────────────────────────────────────────────────
    st.subheader("✨ Your Taste Profile")
    with st.spinner("Generating taste profile…"):
        profile = _load_taste_profile(customer_id, live_mode)
    st.info(profile)

    st.divider()

    # ── Purchase history chart ─────────────────────────────────────────────────
    st.subheader("Top categories in your history")
    try:
        transactions = load_table(PROCESSED_DIR / "transactions")
        articles = load_table(PROCESSED_DIR / "articles").set_index("article_id")
        history = transactions[transactions["customer_id"] == customer_id]
        if not history.empty:
            cats = (
                history["article_id"]
                .map(articles["product_type_name"].to_dict())
                .value_counts()
                .head(10)
                .reset_index()
            )
            cats.columns = ["Category", "Count"]
            fig = px.bar(
                cats,
                x="Count",
                y="Category",
                orientation="h",
                color="Count",
                color_continuous_scale="Blues",
                template="plotly_white",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No purchase history found for this customer.")
    except FileNotFoundError:
        st.caption("Processed data not found. Run preprocessing first.")
