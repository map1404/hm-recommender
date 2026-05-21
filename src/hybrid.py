"""
hybrid.py — Fuwei
Combines ALS collaborative-filtering scores with content-based cosine
similarity into a single ranked list.

Score = alpha * als_score + (1 - alpha) * content_score
Both score arrays are min-max normalised before combining.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.content_based import user_vector, cosine_scores
from src.demo_artifacts import DemoALS
from src.popularity import is_cold_start_customer, recommend_popular
from src.storage import load_table

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

ALPHA = 0.6          # weight on CF scores; 1-ALPHA on content scores
TOP_N = 50           # candidate pool before re-ranking
HF_IMAGE_BASE_URL = (
    "https://huggingface.co/datasets/"
    "einrafh/hnm-fashion-recommendations-data/resolve/main/data/raw/images"
)


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _image_url(article_id: str) -> str:
    article_id = str(article_id)
    return f"{HF_IMAGE_BASE_URL}/{article_id[:3]}/{article_id}.jpg"


def _recommend_cf(model, uid: int, user_items, n: int, filter_purchased: bool):
    try:
        return model.recommend(
            uid,
            user_items,
            N=n,
            filter_already_liked_items=filter_purchased,
        )
    except TypeError:
        return model.recommend(
            uid,
            user_items,
            N=n,
            filter_already_liked=filter_purchased,
        )


def _content_only_recommendations(
    customer_id: str,
    top_k: int,
    article_embeddings: np.ndarray,
    article_idx: pd.DataFrame,
    articles: pd.DataFrame,
    filter_purchased: bool,
) -> list[dict]:
    a_to_idx = dict(zip(article_idx["article_id"], article_idx["article_idx"]))
    idx_to_a = dict(zip(article_idx["article_idx"], article_idx["article_id"]))

    uv = user_vector(customer_id, article_embeddings)
    content_scores = cosine_scores(uv, article_embeddings)

    purchased_ids = set()
    if filter_purchased:
        counts = load_table(PROCESSED_DIR / "user_item_counts")
        purchased_ids = set(counts[counts["customer_id"] == customer_id]["article_id"])

    ranked_positions = np.argsort(-content_scores)
    results = []
    for article_pos in ranked_positions:
        aid = idx_to_a[article_pos]
        if filter_purchased and aid in purchased_ids:
            continue
        row = articles.loc[aid] if aid in articles.index else {}
        results.append({
            "article_id": aid,
            "prod_name": row.get("prod_name", aid),
            "product_type_name": row.get("product_type_name", ""),
            "colour_group_name": row.get("colour_group_name", ""),
            "garment_group_name": row.get("garment_group_name", ""),
            "price": row.get("price", None),
            "image_url": row.get("image_url", _image_url(aid)),
            "hybrid_score": float(content_scores[article_pos]),
        })
        if len(results) >= top_k:
            break
    return results


def recommend(
    customer_id: str,
    top_k: int = 10,
    alpha: float = ALPHA,
    filter_purchased: bool = True,
) -> list[dict]:
    """
    Returns a ranked list of top_k article dicts with hybrid scores.
    Each dict has keys: article_id, product_type_name, colour_group_name,
                        prod_name, price, image_url, hybrid_score.
    """
    if is_cold_start_customer(customer_id):
        return recommend_popular(top_k=top_k)

    # --- load artefacts ---
    user_item = load_npz(MODELS_DIR / "user_item_matrix.npz")
    article_embeddings = np.load(MODELS_DIR / "article_embeddings.npy")

    customer_idx = load_table(PROCESSED_DIR / "customer_index")
    article_idx = load_table(PROCESSED_DIR / "article_index")
    articles = load_table(PROCESSED_DIR / "articles").set_index("article_id")

    c_to_idx = dict(zip(customer_idx["customer_id"], customer_idx["customer_idx"]))
    idx_to_a = dict(zip(article_idx["article_idx"], article_idx["article_id"]))

    if customer_id not in c_to_idx:
        return recommend_popular(top_k=top_k)
    uid = c_to_idx[customer_id]

    try:
        with open(MODELS_DIR / "als_model.pkl", "rb") as f:
            model = pickle.load(f)
    except ModuleNotFoundError:
        return _content_only_recommendations(
            customer_id,
            top_k,
            article_embeddings,
            article_idx,
            articles,
            filter_purchased,
        )

    # --- CF scores ---
    user_row = user_item[uid]
    cf_ids, cf_raw = _recommend_cf(
        model, uid, user_row, TOP_N, filter_purchased
    )
    cf_article_ids = [idx_to_a[i] for i in cf_ids]
    cf_scores = _minmax(np.array(cf_raw))

    # Map CF candidates to article_idx positions
    a_to_idx = dict(zip(article_idx["article_id"], article_idx["article_idx"]))
    cf_positions = [a_to_idx[a] for a in cf_article_ids]

    # --- Content scores (only for CF candidates) ---
    uv = user_vector(customer_id, article_embeddings)
    all_content = cosine_scores(uv, article_embeddings)
    content_raw = all_content[cf_positions]
    content_scores = _minmax(content_raw)

    # --- Hybrid score ---
    hybrid = alpha * cf_scores + (1 - alpha) * content_scores
    ranked_idx = np.argsort(-hybrid)[:top_k]

    results = []
    for rank_pos in ranked_idx:
        aid = cf_article_ids[rank_pos]
        row = articles.loc[aid] if aid in articles.index else {}
        results.append({
            "article_id": aid,
            "prod_name": row.get("prod_name", aid),
            "product_type_name": row.get("product_type_name", ""),
            "colour_group_name": row.get("colour_group_name", ""),
            "garment_group_name": row.get("garment_group_name", ""),
            "price": row.get("price", None),
            "image_url": row.get("image_url", _image_url(aid)),
            "hybrid_score": float(hybrid[rank_pos]),
        })
    return results
