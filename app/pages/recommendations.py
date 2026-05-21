"""
recommendations.py 
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

# ── Global Shopping Cart Init ──────────────────────────────────────────
if "cart" not in st.session_state:
    st.session_state["cart"] = []

# ── Custom CSS for Professional Storefront ────────────────────────────
st.markdown("""
<style>
    .product-card { 
        background: white; 
        padding: 0px; 
        transition: transform 0.2s; 
    }
    .product-card:hover { transform: translateY(-3px); }
    
    /* H&M Style Black Button */
    .hm-button {
        background-color: #222222;
        color: #ffffff;
        border: none;
        padding: 12px;
        width: 100%;
        text-align: center;
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        cursor: pointer;
        display: block;
        margin-top: 10px;
    }
    .hm-button:hover { background-color: #E50010; }
</style>
""", unsafe_allow_html=True)

def _load_recommendations(customer_id: str, live: bool, cold_start: bool = False) -> list[dict]:
    from src.hybrid import recommend
    from src.popularity import is_cold_start_customer, recommend_popular

    if cold_start or is_cold_start_customer(customer_id):
        return recommend_popular(top_k=TOP_K)

    if live:
        from src.llm import generate_taste_profile, generate_explanation
        taste_profile = generate_taste_profile(customer_id)
        articles = recommend(customer_id, top_k=TOP_K)
        for a in articles:
            a["explanation"] = generate_explanation(customer_id, taste_profile, a)
        return articles

    explanations_path = DEMO_CACHE_DIR / "explanations.json"
    all_explanations = json.loads(explanations_path.read_text()) if explanations_path.exists() else {}
    cust_explanations = all_explanations.get(customer_id, {})

    articles = recommend(customer_id, top_k=TOP_K)
    for a in articles:
        a["explanation"] = cust_explanations.get(a["article_id"], "")
    return articles

def _article_card(article: dict):
    with st.container():
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        # Image
        st.image(article.get("image_url", ""), use_container_width=True)
        
        # Product Info
        st.markdown(f"<p style='font-size:14px; margin:5px 0 0 0;'>{article.get('prod_name', 'Unknown')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:12px; color:#777; margin:0;'>{article.get('product_type_name', '')}</p>", unsafe_allow_html=True)
        
        price = article.get("price", 0)
        st.markdown(f"<p style='font-weight:700; margin:5px 0;'>£{price:.2f}</p>", unsafe_allow_html=True)
        
        
        if st.button("ADD TO CART", key=f"btn_{article.get('article_id')}"):
            st.session_state["cart"].append(article)
            st.toast(f"Added to bag", icon="🛍️")
            
        # Style Note
        explanation = article.get("explanation", "")
        if explanation:
            st.markdown(
                f"""<div style="background:#F7F7F7; border-left:3px solid #E50010;
                padding:10px; font-size:11px; color:#444; margin-top:15px; line-height:1.4;">
                <span style="font-weight:bold;">💡 STYLE NOTE:</span> {explanation}</div>""",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

def render():
    if "cart" not in st.session_state:
        st.session_state["cart"] = []
        
    customer_id = st.session_state.get("customer_id", "")
    live_mode = st.session_state.get("live_mode", False)
    cold_start = st.session_state.get("cold_start", False)

    if not cold_start and not customer_id:
        return

    # Header section with Cart counter
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h2 style='font-weight:300; margin-bottom:30px;'>RECOMMENDED FOR YOU</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**BAG ({len(st.session_state['cart'])})**", unsafe_allow_html=True)

    with st.spinner("Curating your collection..."):
        try:
            recs = _load_recommendations(customer_id, live_mode, cold_start=cold_start)
        except Exception:
            return

    # Grid Rendering
    for row_start in range(0, len(recs), COLS):
        cols = st.columns(COLS)
        for col_idx, article in enumerate(recs[row_start: row_start + COLS]):
            with cols[col_idx]:
                _article_card(article)
                st.write("")