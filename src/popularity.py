"""
popularity.py
Cold-start popularity-based recommendation fallback.

When a customer has no purchase history (new user, guest, or an unknown ID),
the personalized hybrid pipeline cannot produce meaningful scores. This module
ranks articles by recent purchase volume + unique customer reach and returns
records compatible with the existing UI.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from src.storage import load_table

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
MODELS_DIR = Path("models")

RECENT_DAYS = 60
TOP_N_SAVED = 200

COLD_START_IDS = {None, "", "new_user", "cold_start", "guest", "unknown"}

HF_IMAGE_BASE_URL = (
    "https://huggingface.co/datasets/"
    "einrafh/hnm-fashion-recommendations-data/resolve/main/data/raw/images"
)


def _image_url(article_id: str) -> str:
    article_id = str(article_id)
    return f"{HF_IMAGE_BASE_URL}/{article_id[:3]}/{article_id}.jpg"


def _load_transactions() -> pd.DataFrame:
    processed_pkl = PROCESSED_DIR / "transactions.pkl"
    processed_parquet = PROCESSED_DIR / "transactions.parquet"
    if processed_pkl.exists() or processed_parquet.exists():
        df = load_table(PROCESSED_DIR / "transactions")
    else:
        raw_path = RAW_DIR / "transactions_train.csv"
        if not raw_path.exists():
            raise FileNotFoundError(
                "No transactions found. Expected one of "
                f"{processed_pkl}, {processed_parquet}, or {raw_path}."
            )
        df = pd.read_csv(raw_path)
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    return df


def _load_articles() -> pd.DataFrame:
    processed_pkl = PROCESSED_DIR / "articles.pkl"
    processed_parquet = PROCESSED_DIR / "articles.parquet"
    if processed_pkl.exists() or processed_parquet.exists():
        return load_table(PROCESSED_DIR / "articles")
    raw_path = RAW_DIR / "articles.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path, dtype={"article_id": str})
    raise FileNotFoundError(
        "No articles file found in data/processed or data/raw."
    )


def _explanation(row: pd.Series) -> str:
    name = row.get("prod_name") or row.get("article_id")
    cat = row.get("product_type_name") or "item"
    count = int(row.get("purchase_count", 0))
    customers = int(row.get("unique_customers", 0))
    return (
        f"Trending {cat.lower()} — {name} was bought {count} times by "
        f"{customers} different shoppers in the last {RECENT_DAYS} days."
    )


def build_popularity_recommendations(
    recent_days: int = RECENT_DAYS,
    top_n: int = TOP_N_SAVED,
) -> pd.DataFrame:
    """Compute the popularity ranking and persist it to models/."""
    transactions = _load_transactions()
    articles = _load_articles()

    cutoff = transactions["t_dat"].max() - pd.Timedelta(days=recent_days)
    recent = transactions[transactions["t_dat"] >= cutoff]

    grouped = (
        recent.groupby("article_id")
        .agg(
            purchase_count=("customer_id", "size"),
            unique_customers=("customer_id", "nunique"),
        )
        .reset_index()
    )
    grouped["popularity_score"] = (
        0.7 * grouped["purchase_count"] + 0.3 * grouped["unique_customers"]
    )

    if "article_id" in articles.columns:
        articles = articles.copy()
        article_id_type = type(grouped["article_id"].iloc[0]) if len(grouped) else str
        articles["article_id"] = articles["article_id"].astype(article_id_type)

    merged = grouped.merge(articles, on="article_id", how="left")
    merged = (
        merged.sort_values("popularity_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = MODELS_DIR / "popular_items.parquet"
    pkl_path = MODELS_DIR / "popular_items.pkl"
    try:
        merged.to_parquet(parquet_path, index=False)
        print(f"Saved {len(merged)} popular items → {parquet_path}")
    except (ImportError, OSError, ValueError) as e:
        with open(pkl_path, "wb") as f:
            pickle.dump(merged, f)
        print(f"Parquet failed ({e}); saved → {pkl_path}")

    return merged


def _load_popular_items() -> pd.DataFrame:
    parquet_path = MODELS_DIR / "popular_items.parquet"
    pkl_path = MODELS_DIR / "popular_items.pkl"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return build_popularity_recommendations()


def recommend_popular(top_k: int = 10) -> list[dict]:
    """Return top_k popularity-based recommendations as UI-compatible dicts."""
    popular = _load_popular_items().head(top_k)

    results: list[dict] = []
    for _, row in popular.iterrows():
        aid = str(row["article_id"])
        results.append({
            "article_id": aid,
            "prod_name": row.get("prod_name", aid),
            "product_type_name": row.get("product_type_name", ""),
            "colour_group_name": row.get("colour_group_name", ""),
            "garment_group_name": row.get("garment_group_name", ""),
            "price": row.get("price", None),
            "image_url": row.get("image_url", _image_url(aid)),
            "hybrid_score": float(row.get("popularity_score", 0.0)),
            "popularity_score": float(row.get("popularity_score", 0.0)),
            "purchase_count": int(row.get("purchase_count", 0)),
            "unique_customers": int(row.get("unique_customers", 0)),
            "recommendation_type": "cold_start_popularity",
            "explanation": _explanation(row),
        })
    return results


def is_cold_start_customer(customer_id: Optional[str]) -> bool:
    """True if the customer should fall back to popularity recommendations."""
    if customer_id is None:
        return True
    cid = customer_id.strip().lower() if isinstance(customer_id, str) else customer_id
    if cid in COLD_START_IDS:
        return True

    try:
        customer_idx = load_table(PROCESSED_DIR / "customer_index")
    except FileNotFoundError:
        return True
    if customer_id not in set(customer_idx["customer_id"]):
        return True

    try:
        counts = load_table(PROCESSED_DIR / "user_item_counts")
    except FileNotFoundError:
        return False
    has_history = (counts["customer_id"] == customer_id).any()
    return not bool(has_history)


if __name__ == "__main__":
    build_popularity_recommendations()
