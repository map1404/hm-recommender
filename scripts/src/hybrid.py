"""
hybrid.py
---------
ML Developer B (Fuwei) module.

Combines ALS collaborative-filtering scores with content-based cosine
similarity into a single ranked list.

Score = alpha * cf_score + (1 - alpha) * content_score
Both score arrays are min-max normalised before combining.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.content_based import user_vector, cosine_scores
from src.storage import load_table

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

ALPHA = 0.6
TOP_N = 50

HF_IMAGE_BASE = (
    "https://huggingface.co/datasets/"
    "einrafh/hnm-fashion-recommendations-data/resolve/main/data/raw/images"
)


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _image_url(article_id: str) -> str:
    aid = str(article_id)
    return f"{HF_IMAGE_BASE}/{aid[:3]}/{aid}.jpg"


def recommend(
    customer_id: str,
    top_k: int = 10,
    alpha: float = ALPHA,
    filter_purchased: bool = True,
) -> list[dict]:
    """
    Returns a ranked list of top_k article dicts with hybrid scores.
    Each dict has: article_id, prod_name, product_type_name,
                   colour_group_name, garment_group_name,
                   price, image_url, hybrid_score.
    """
    user_item = load_npz(MODELS_DIR / "user_item_matrix.npz")
    article_embeddings = np.load(MODELS_DIR / "article_embeddings.npy")

    customer_idx = load_table(PROCESSED_DIR / "customer_index")
    article_idx = load_table(PROCESSED_DIR / "article_index")
    articles = load_table(PROCESSED_DIR / "articles").set_index("article_id")

    c_to_idx = dict(zip(customer_idx["customer_id"], customer_idx["customer_idx"]))
    idx_to_a = dict(zip(article_idx["article_idx"], article_idx["article_id"]))
    a_to_idx = dict(zip(article_idx["article_id"], article_idx["article_idx"]))

    if customer_id not in c_to_idx:
        raise ValueError(f"Unknown customer ID: {customer_id}")
    uid = c_to_idx[customer_id]

    with open(MODELS_DIR / "als_model.pkl", "rb") as f:
        model = pickle.load(f)

    # CF scores
    user_row = user_item[uid]
    try:
        cf_ids, cf_raw = model.recommend(
            uid, user_row, N=TOP_N, filter_already_liked_items=filter_purchased
        )
    except TypeError:
        cf_ids, cf_raw = model.recommend(
            uid, user_row, N=TOP_N, filter_already_liked=filter_purchased
        )

    cf_article_ids = [idx_to_a[i] for i in cf_ids]
    cf_scores = _minmax(np.array(cf_raw))

    # Content scores (only for CF candidates)
    uv = user_vector(customer_id, article_embeddings)
    all_content = cosine_scores(uv, article_embeddings)
    cf_positions = [a_to_idx[a] for a in cf_article_ids if a in a_to_idx]
    content_raw = all_content[cf_positions]
    content_scores_norm = _minmax(content_raw)

    # Hybrid
    hybrid = alpha * cf_scores[:len(cf_positions)] + (1 - alpha) * content_scores_norm
    ranked_idx = np.argsort(-hybrid)[:top_k]

    results = []
    for rank_pos in ranked_idx:
        aid = cf_article_ids[rank_pos]
        row = articles.loc[aid] if aid in articles.index else {}
        results.append({
            "article_id": aid,
            "prod_name": row.get("prod_name", aid) if hasattr(row, "get") else aid,
            "product_type_name": row.get("product_type_name", "") if hasattr(row, "get") else "",
            "colour_group_name": row.get("colour_group_name", "") if hasattr(row, "get") else "",
            "garment_group_name": row.get("garment_group_name", "") if hasattr(row, "get") else "",
            "price": row.get("price", None) if hasattr(row, "get") else None,
            "image_url": _image_url(aid),
            "hybrid_score": float(hybrid[rank_pos]),
        })
    return results
