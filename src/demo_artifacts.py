"""
demo_artifacts.py

Creates a small self-contained demo dataset so the Streamlit app can run
without the Kaggle download or model training pipeline.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from src.storage import save_table

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
CACHE_DIR = Path("demo_cache")


@dataclass
class DemoALS:
    """Small stand-in for the ALS API used by the lightweight demo dataset."""

    recommendations_by_user: dict[int, tuple[np.ndarray, np.ndarray]]

    def recommend(self, userid, _user_items, n=10, _filter_already_liked=True):
        """Return the precomputed top-N recommendations for a demo user."""
        item_ids, scores = self.recommendations_by_user[userid]
        return item_ids[:n], scores[:n]


def _norm(vec):
    arr = np.array(vec, dtype=np.float32)
    denom = np.linalg.norm(arr)
    return arr / denom if denom else arr


def _articles():
    return [
        {
            "article_id": "100000001",
            "prod_name": "Soft Beige Knit Cardigan",
            "product_type_name": "Cardigan",
            "colour_group_name": "Beige",
            "garment_group_name": "Knitwear",
            "price": 39.99,
            "image_url": "https://picsum.photos/seed/hm-100000001/600/800",
            "text": "cardigan | knitwear | beige | soft layers | casual minimal",
            "embedding": _norm([0.95, 0.20, 0.10, 0.85, 0.10, 0.10]),
        },
        {
            "article_id": "100000002",
            "prod_name": "White Relaxed Cotton Shirt",
            "product_type_name": "Shirt",
            "colour_group_name": "White",
            "garment_group_name": "Blouses",
            "price": 29.99,
            "image_url": "https://picsum.photos/seed/hm-100000002/600/800",
            "text": "shirt | white | crisp basics | relaxed silhouette",
            "embedding": _norm([0.85, 0.15, 0.10, 0.75, 0.20, 0.10]),
        },
        {
            "article_id": "100000003",
            "prod_name": "Taupe Wide-Leg Trousers",
            "product_type_name": "Trousers",
            "colour_group_name": "Taupe",
            "garment_group_name": "Trousers",
            "price": 44.99,
            "image_url": "https://picsum.photos/seed/hm-100000003/600/800",
            "text": "trousers | taupe | tailored | wide leg | clean lines",
            "embedding": _norm([0.90, 0.10, 0.15, 0.70, 0.35, 0.10]),
        },
        {
            "article_id": "100000004",
            "prod_name": "Black Ribbed Midi Dress",
            "product_type_name": "Dress",
            "colour_group_name": "Black",
            "garment_group_name": "Dresses",
            "price": 49.99,
            "image_url": "https://picsum.photos/seed/hm-100000004/600/800",
            "text": "dress | black | ribbed | fitted | evening minimal",
            "embedding": _norm([0.70, 0.10, 0.20, 0.80, 0.65, 0.10]),
        },
        {
            "article_id": "100000005",
            "prod_name": "Charcoal Tailored Blazer",
            "product_type_name": "Blazer",
            "colour_group_name": "Grey",
            "garment_group_name": "Jackets",
            "price": 59.99,
            "image_url": "https://picsum.photos/seed/hm-100000005/600/800",
            "text": "blazer | charcoal | tailored | sharp | polished layers",
            "embedding": _norm([0.75, 0.15, 0.15, 0.65, 0.85, 0.10]),
        },
        {
            "article_id": "100000006",
            "prod_name": "Navy Pleated Skirt",
            "product_type_name": "Skirt",
            "colour_group_name": "Blue",
            "garment_group_name": "Skirts",
            "price": 34.99,
            "image_url": "https://picsum.photos/seed/hm-100000006/600/800",
            "text": "skirt | navy | pleated | polished | office casual",
            "embedding": _norm([0.65, 0.20, 0.15, 0.60, 0.75, 0.15]),
        },
        {
            "article_id": "100000007",
            "prod_name": "Heather Grey Oversized Hoodie",
            "product_type_name": "Hoodie",
            "colour_group_name": "Grey",
            "garment_group_name": "Jersey Basic",
            "price": 32.99,
            "image_url": "https://picsum.photos/seed/hm-100000007/600/800",
            "text": "hoodie | grey | oversized | sporty | relaxed basics",
            "embedding": _norm([0.20, 0.90, 0.20, 0.25, 0.10, 0.85]),
        },
        {
            "article_id": "100000008",
            "prod_name": "Black Technical Leggings",
            "product_type_name": "Leggings",
            "colour_group_name": "Black",
            "garment_group_name": "Trousers",
            "price": 27.99,
            "image_url": "https://picsum.photos/seed/hm-100000008/600/800",
            "text": "leggings | black | technical | sporty | fitted",
            "embedding": _norm([0.15, 0.95, 0.20, 0.20, 0.10, 0.90]),
        },
        {
            "article_id": "100000009",
            "prod_name": "White Running Tank",
            "product_type_name": "Vest top",
            "colour_group_name": "White",
            "garment_group_name": "Jersey Fancy",
            "price": 19.99,
            "image_url": "https://picsum.photos/seed/hm-100000009/600/800",
            "text": "tank | white | running | breathable | activewear",
            "embedding": _norm([0.10, 0.88, 0.25, 0.25, 0.10, 0.80]),
        },
        {
            "article_id": "100000010",
            "prod_name": "Sage Green Puffer Vest",
            "product_type_name": "Waistcoat",
            "colour_group_name": "Green",
            "garment_group_name": "Outdoor",
            "price": 46.99,
            "image_url": "https://picsum.photos/seed/hm-100000010/600/800",
            "text": "puffer vest | green | sporty outerwear | casual",
            "embedding": _norm([0.15, 0.82, 0.30, 0.20, 0.15, 0.78]),
        },
        {
            "article_id": "100000011",
            "prod_name": "Rust Floral Wrap Dress",
            "product_type_name": "Dress",
            "colour_group_name": "Orange",
            "garment_group_name": "Dresses",
            "price": 54.99,
            "image_url": "https://picsum.photos/seed/hm-100000011/600/800",
            "text": "wrap dress | rust | floral | feminine | soft drape",
            "embedding": _norm([0.55, 0.15, 0.95, 0.40, 0.35, 0.15]),
        },
        {
            "article_id": "100000012",
            "prod_name": "Cream Satin Blouse",
            "product_type_name": "Blouse",
            "colour_group_name": "Cream",
            "garment_group_name": "Blouses",
            "price": 41.99,
            "image_url": "https://picsum.photos/seed/hm-100000012/600/800",
            "text": "satin blouse | cream | elegant | draped | feminine",
            "embedding": _norm([0.60, 0.15, 0.80, 0.55, 0.45, 0.10]),
        },
        {
            "article_id": "100000013",
            "prod_name": "Berry Pleated Midi Skirt",
            "product_type_name": "Skirt",
            "colour_group_name": "Red",
            "garment_group_name": "Skirts",
            "price": 38.99,
            "image_url": "https://picsum.photos/seed/hm-100000013/600/800",
            "text": "midi skirt | berry | pleated | feminine | occasion",
            "embedding": _norm([0.50, 0.15, 0.88, 0.45, 0.40, 0.10]),
        },
        {
            "article_id": "100000014",
            "prod_name": "Black Heeled Ankle Boots",
            "product_type_name": "Boots",
            "colour_group_name": "Black",
            "garment_group_name": "Shoes",
            "price": 69.99,
            "image_url": "https://picsum.photos/seed/hm-100000014/600/800",
            "text": "ankle boots | black | sleek | elevated styling",
            "embedding": _norm([0.55, 0.10, 0.65, 0.45, 0.55, 0.10]),
        },
        {
            "article_id": "100000015",
            "prod_name": "Soft Pink Knit Top",
            "product_type_name": "Top",
            "colour_group_name": "Pink",
            "garment_group_name": "Knitwear",
            "price": 24.99,
            "image_url": "https://picsum.photos/seed/hm-100000015/600/800",
            "text": "knit top | pink | soft | feminine | fitted",
            "embedding": _norm([0.48, 0.12, 0.82, 0.50, 0.30, 0.10]),
        },
    ]


def _customers():
    return [
        {
            "customer_id": "demo-minimal-neutral",
            "age": 29,
            "club_member_status": "ACTIVE",
        },
        {
            "customer_id": "demo-sporty-urban",
            "age": 34,
            "club_member_status": "ACTIVE",
        },
        {
            "customer_id": "demo-feminine-polish",
            "age": 27,
            "club_member_status": "ACTIVE",
        },
    ]


def _profiles():
    return {
        "demo-minimal-neutral": (
            "You prefer clean, minimal outfits built around soft neutrals, knit layers, "
            "and relaxed tailoring. Your wardrobe leans toward versatile basics with "
            "polished silhouettes rather than loud prints or bright colour."
        ),
        "demo-sporty-urban": (
            "You gravitate toward sporty essentials in black, grey, white, and muted green. "
            "You tend to choose practical activewear shapes like hoodies, leggings, and "
            "technical layers with a relaxed, functional feel."
        ),
        "demo-feminine-polish": (
            "You prefer feminine pieces with fluid shapes, soft shine, and richer accent colours. "
            "Your style mixes dresses, blouses, and skirts that feel polished and slightly dressy "
            "without becoming overly formal."
        ),
    }


def _transactions():
    return [
        ("demo-minimal-neutral", "100000001", "2025-03-01"),
        ("demo-minimal-neutral", "100000002", "2025-03-08"),
        ("demo-minimal-neutral", "100000003", "2025-03-14"),
        ("demo-minimal-neutral", "100000001", "2025-03-28"),
        ("demo-minimal-neutral", "100000004", "2025-04-02"),
        ("demo-minimal-neutral", "100000003", "2025-04-12"),
        ("demo-sporty-urban", "100000007", "2025-03-03"),
        ("demo-sporty-urban", "100000008", "2025-03-11"),
        ("demo-sporty-urban", "100000009", "2025-03-21"),
        ("demo-sporty-urban", "100000007", "2025-03-30"),
        ("demo-sporty-urban", "100000010", "2025-04-04"),
        ("demo-sporty-urban", "100000008", "2025-04-18"),
        ("demo-feminine-polish", "100000011", "2025-03-06"),
        ("demo-feminine-polish", "100000012", "2025-03-16"),
        ("demo-feminine-polish", "100000013", "2025-03-25"),
        ("demo-feminine-polish", "100000015", "2025-04-01"),
        ("demo-feminine-polish", "100000011", "2025-04-10"),
        ("demo-feminine-polish", "100000014", "2025-04-20"),
    ]


def _explanation(article, theme):
    return (
        f"This {article['product_type_name'].lower()} fits your {theme} style through its "
        f"{article['colour_group_name'].lower()} tone and {article['garment_group_name'].lower()} silhouette."
    )


def build_demo_artifacts():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    customers = pd.DataFrame(_customers())
    articles_raw = _articles()
    articles = pd.DataFrame([{k: v for k, v in row.items() if k != "embedding"} for row in articles_raw])
    embeddings = np.stack([row["embedding"] for row in articles_raw]).astype(np.float32)

    tx = pd.DataFrame(_transactions(), columns=["customer_id", "article_id", "t_dat"])
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])

    counts = tx.groupby(["customer_id", "article_id"]).size().reset_index(name="count")
    customer_index = pd.DataFrame(
        {"customer_id": customers["customer_id"], "customer_idx": range(len(customers))}
    )
    article_index = pd.DataFrame(
        {"article_id": articles["article_id"], "article_idx": range(len(articles))}
    )

    c_to_idx = dict(zip(customer_index["customer_id"], customer_index["customer_idx"]))
    a_to_idx = dict(zip(article_index["article_id"], article_index["article_idx"]))
    matrix = csr_matrix(
        (
            counts["count"].to_numpy(),
            (
                counts["customer_id"].map(c_to_idx).to_numpy(),
                counts["article_id"].map(a_to_idx).to_numpy(),
            ),
        ),
        shape=(len(customers), len(articles)),
    )
    save_npz(MODELS_DIR / "user_item_matrix.npz", matrix)

    themes = {
        "demo-minimal-neutral": "minimal-neutral",
        "demo-sporty-urban": "sporty-functional",
        "demo-feminine-polish": "feminine-polished",
    }
    recs = {
        "demo-minimal-neutral": ["100000005", "100000012", "100000006", "100000014", "100000011"],
        "demo-sporty-urban": ["100000010", "100000009", "100000008", "100000005", "100000014"],
        "demo-feminine-polish": ["100000012", "100000013", "100000015", "100000014", "100000004"],
    }

    demo_model = DemoALS(
        recommendations_by_user={
            c_to_idx[cid]: (
                np.array([a_to_idx[aid] for aid in article_ids], dtype=np.int64),
                np.array([1.0 - 0.05 * i for i in range(len(article_ids))], dtype=np.float32),
            )
            for cid, article_ids in recs.items()
        }
    )
    with open(MODELS_DIR / "als_model.pkl", "wb") as fh:
        pickle.dump(demo_model, fh)
    np.save(MODELS_DIR / "article_embeddings.npy", embeddings)

    save_table(customers, DATA_DIR / "customers")
    save_table(articles, DATA_DIR / "articles")
    save_table(tx, DATA_DIR / "transactions")
    save_table(counts, DATA_DIR / "user_item_counts")
    save_table(customer_index, DATA_DIR / "customer_index")
    save_table(article_index, DATA_DIR / "article_index")

    profiles = _profiles()
    explanations = {}
    article_rows = {row["article_id"]: row for row in articles_raw}
    for customer_id, article_ids in recs.items():
        explanations[customer_id] = {
            article_id: _explanation(article_rows[article_id], themes[customer_id])
            for article_id in article_ids
        }

    (CACHE_DIR / "taste_profiles.json").write_text(json.dumps(profiles, indent=2))
    (CACHE_DIR / "explanations.json").write_text(json.dumps(explanations, indent=2))
    print("Demo artifacts generated in data/processed, models, and demo_cache.")


if __name__ == "__main__":
    build_demo_artifacts()
