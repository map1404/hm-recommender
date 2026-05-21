"""
recommendations.py
------------------
Frontend Developer (Lei) module.

Recommendations page:
  - Taste profile recap banner
  - 5-column grid of recommended items
  - Real H&M product images from HuggingFace CDN
  - Per-item explanation badge
"""

import json
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEMO_CACHE_DIR = ROOT_DIR / "demo_cache"
TOP_K = 10
COLS = 5


@st.cache_data
def _load_cache():
    profiles, explanations, recs = {}, {}, {}
    for name, store in [
        ("taste_profiles.json", profiles),
        ("explanations.json", explanations),
        ("recommendations.json", recs),
    ]:
        path = DEMO_CACHE_DIR / name
        if path.exists():
            store.update(json.loads(path.read_text()))
    return profiles, explanations, recs


def _article_card(article: dict):
    with st.container(border=True):
        image_url = article.get("image_url", "")
        if image_url:
            st.image(image_url, use_container_width=True)
        else:
            # Colour tile fallback
            colour = (article.get("colour_group_name") or "").lower()
            bg = (
                "#f7c5c5" if "pink" in colour or "rose" in colour else
                "#c5d4f7" if "blue" in colour or "navy" in colour else
                "#c5f7d4" if "green" in colour else
                "#f7f0c5" if "yellow" in colour else
                "#d0d0d0" if "black" in colour else
                "#f5f5f0" if "white" in colour or "cream" in colour else
                "#e0e0e0"
            )
            initials = (article.get("prod_name") or "??")[:2].upper()
            st.markdown(
                f'<div style="background:{bg};height:160px;border-radius:8px;'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:1.8rem;font-weight:bold;color:#555;margin-bottom:0.5rem;">'
                f'{initials}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"**{article.get('prod_name', 'Item')}**")

        tags = ""
        if article.get("product_type_name"):
            tags += f'<span class="cat-tag">{article["product_type_name"]}</span>'
        if article.get("colour_group_name"):
            tags += f'<span class="cat-tag">🎨 {article["colour_group_name"]}</span>'
        if tags:
            st.markdown(tags, unsafe_allow_html=True)

        price = article.get("price")
        if price:
            try:
                st.markdown(
                    f'<span class="score-pill">€{float(price):.2f}</span>',
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        explanation = article.get("explanation", "")
        if explanation:
            st.markdown(
                f'<div class="explanation-badge">💡 {explanation}</div>',
                unsafe_allow_html=True,
            )


def _load_recommendations_live(customer_id: str) -> tuple[str, list]:
    from src.llm import generate_taste_profile, explain_recommendation
    from src.hybrid import recommend

    taste_profile = generate_taste_profile(customer_id)
    articles = recommend(customer_id, top_k=TOP_K)
    for a in articles:
        a["explanation"] = explain_recommendation(taste_profile, a)
    return taste_profile, articles


def render():
    customer_id = st.session_state.get("customer_id", "")
    live_mode = st.session_state.get("live_mode", False)

    st.markdown("""
    <div class="brand-header">
        <h1>✨ Your Recommendations</h1>
        <p>Top 10 picks curated for your unique style</p>
    </div>
    """, unsafe_allow_html=True)

    if not customer_id:
        st.info("Select a customer from the sidebar to begin.")
        return

    if live_mode:
        with st.spinner("🤖 Generating personalised recommendations..."):
            try:
                taste_profile, articles = _load_recommendations_live(customer_id)
            except Exception as e:
                st.error(str(e))
                return
    else:
        profiles, explanations, recs_cache = _load_cache()
        taste_profile = profiles.get(customer_id, "")
        cached_recs = recs_cache.get(customer_id, [])
        cust_exps = explanations.get(customer_id, {})

        if not cached_recs:
            st.warning(
                "No cached recommendations found. "
                "Run `python src/inference.py --cache` or enable Live LLM mode."
            )
            return

        articles = []
        for rec in cached_recs:
            rec["explanation"] = cust_exps.get(rec.get("article_id", ""), "")
            articles.append(rec)

    if taste_profile:
        st.markdown(
            f'<div class="taste-profile-box"><strong>Your Style:</strong> {taste_profile}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"### Recommended for you ({len(articles)} items)")
    st.markdown("---")

    for row_start in range(0, len(articles), COLS):
        cols = st.columns(COLS)
        for col, article in zip(cols, articles[row_start:row_start + COLS]):
            with col:
                _article_card(article)
        st.markdown("<br>", unsafe_allow_html=True)
