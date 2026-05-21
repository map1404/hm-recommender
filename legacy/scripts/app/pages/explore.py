"""
explore.py
----------
Frontend Developer (Lei) module.

Explore Styles page:
  - Groups demo customers into style archetypes
  - Shows taste profile + top picks per customer
  - Archetype distribution chart
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEMO_CACHE_DIR = ROOT_DIR / "demo_cache"

ARCHETYPES = {
    "🏃 Casual":     ["casual", "jersey", "jogger", "hoodie", "sport", "everyday", "comfort"],
    "👔 Formal":     ["formal", "blazer", "suit", "tailored", "office", "polished"],
    "🌸 Feminine":   ["dress", "feminine", "floral", "skirt", "blouse", "romantic", "wrap"],
    "🖤 Minimalist": ["minimal", "monochromatic", "black", "grey", "clean", "simple", "neutral"],
    "🌈 Bold":       ["bold", "vibrant", "pattern", "print", "bright", "celebration", "colour"],
    "🛋️ Lounge":    ["lounge", "cosy", "knit", "soft", "oversized", "relaxed", "comfortable"],
}


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


def _classify(profile_text: str) -> list:
    tl = profile_text.lower()
    matched = [label for label, kws in ARCHETYPES.items() if any(k in tl for k in kws)]
    return matched or ["🗂️ Other"]


def render():
    st.markdown("""
    <div class="brand-header">
        <h1>🎨 Explore Style Archetypes</h1>
        <p>Browse demo customers grouped by their style DNA</p>
    </div>
    """, unsafe_allow_html=True)

    profiles, explanations, recs = _load_cache()

    if not profiles:
        st.info(
            "No demo cache found. "
            "Run `python src/inference.py --cache` to populate it."
        )
        return

    # Build archetype map
    archetype_map: dict = {}
    for cid, text in profiles.items():
        for label in _classify(text):
            archetype_map.setdefault(label, []).append(cid)

    # Tabs
    tab_labels = list(archetype_map.keys())
    tabs = st.tabs(tab_labels)

    for tab, label in zip(tabs, tab_labels):
        with tab:
            customer_ids = archetype_map[label]
            st.markdown(f"**{len(customer_ids)} customer(s) in this archetype**")

            for idx, cid in enumerate(customer_ids[:6]):
                with st.expander(f"👤 {cid[:24]}{'…' if len(cid) > 24 else ''}"):
                    profile_text = profiles.get(cid, "")
                    if profile_text:
                        st.markdown(
                            f'<div class="taste-profile-box">{profile_text}</div>',
                            unsafe_allow_html=True,
                        )

                    cid_recs = recs.get(cid, [])[:5]
                    cid_exps = explanations.get(cid, {})

                    if cid_recs:
                        st.markdown("**Top picks:**")
                        for item in cid_recs:
                            aid = item.get("article_id", "")
                            name = item.get("prod_name", aid)
                            colour = item.get("colour_group_name", "")
                            exp = cid_exps.get(aid, "")
                            st.markdown(
                                f"• **{name}**" + (f" · {colour}" if colour else "")
                            )
                            if exp:
                                st.markdown(
                                    f'<div class="explanation-badge">💡 {exp}</div>',
                                    unsafe_allow_html=True,
                                )

                    btn_key = f"explore_{tab_labels.index(label)}_{idx}_{cid[-4:]}"
                    if st.button("View full profile →", key=btn_key):
                        st.session_state["customer_id"] = cid
                        st.session_state["current_page"] = "👤 Profile"
                        st.rerun()

    # Distribution chart
    st.markdown("---")
    st.markdown("### 📊 Archetype Distribution")
    dist = pd.DataFrame([
        {"Archetype": label, "Customers": len(ids)}
        for label, ids in archetype_map.items()
    ]).sort_values("Customers", ascending=False)

    fig = px.bar(
        dist, x="Archetype", y="Customers",
        color="Customers",
        color_continuous_scale=["#f9e0e3", "#e63946"],
        text="Customers",
        template="plotly_white",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis_title="",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
