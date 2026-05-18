"""
recommendations.py — Lei
Recommendations page:
  • 2-column grid of recommended items
  • Each item: image, name, category, price, explanation badge
"""

from pathlib import Path
import json

import streamlit as st

DEMO_CACHE_DIR = Path("demo_cache")
TOP_K = 10
COLS = 2


def _load_recommendations(customer_id: str, live: bool) -> list[dict]:
    from src.hybrid import recommend

    if live:
        from src.llm import generate_taste_profile, generate_explanation
        taste_profile = generate_taste_profile(customer_id)
        articles = recommend(customer_id, top_k=TOP_K)
        for a in articles:
            a["explanation"] = generate_explanation(customer_id, taste_profile, a)
        return articles

    # Cached mode
    explanations_path = DEMO_CACHE_DIR / "explanations.json"
    all_explanations = {}
    if explanations_path.exists():
        all_explanations = json.loads(explanations_path.read_text())
    cust_explanations = all_explanations.get(customer_id, {})

    articles = recommend(customer_id, top_k=TOP_K)
    for a in articles:
        a["explanation"] = cust_explanations.get(a["article_id"], "No cached explanation.")
    return articles


def _article_card(article: dict):
    with st.container(border=True):
        st.image(article.get("image_url", ""), use_container_width=True)
        st.markdown(f"**{article.get('prod_name', 'Unknown')}**")
        st.caption(
            f"{article.get('product_type_name', '')} · "
            f"{article.get('colour_group_name', '')} · "
            f"{article.get('garment_group_name', '')}"
        )
        price = article.get("price")
        if price:
            st.markdown(f"£{price:.2f}")

        # Explanation badge
        explanation = article.get("explanation", "")
        if explanation:
            st.markdown(
                f"""<div style="background:#EEF4FF;border-left:3px solid #4A7CF7;
                padding:8px 10px;border-radius:4px;font-size:13px;color:#1a1a2e;
                margin-top:8px">💡 {explanation}</div>""",
                unsafe_allow_html=True,
            )


def render():
    customer_id = st.session_state.get("customer_id", "")
    live_mode = st.session_state.get("live_mode", False)

    if not customer_id:
        st.info("Select a customer from the sidebar to begin.")
        return

    st.header("Recommended for you")
    st.caption(f"Customer: `{customer_id[:20]}…`")

    with st.spinner("Loading recommendations…"):
        try:
            recs = _load_recommendations(customer_id, live_mode)
        except (KeyError, FileNotFoundError) as e:
            st.error(str(e))
            return

    if not recs:
        st.warning("No recommendations found for this customer.")
        return

    # Render in a grid
    for row_start in range(0, len(recs), COLS):
        cols = st.columns(COLS)
        for col_idx, article in enumerate(recs[row_start: row_start + COLS]):
            with cols[col_idx]:
                _article_card(article)
