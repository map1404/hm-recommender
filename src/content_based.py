"""
content_based.py — Fuwei
Builds article text embeddings using sentence-transformers and computes
cosine similarity between a user vector and all article embeddings.

Outputs:
  models/article_embeddings.npy  — (n_articles, embedding_dim) float32 array
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.storage import load_table

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 512


def build_embeddings():
    from sentence_transformers import SentenceTransformer

    print(f"Loading article metadata...")
    articles = load_table(PROCESSED_DIR / "articles")
    article_idx = load_table(PROCESSED_DIR / "article_index")

    # Align articles to the same order as the matrix column index
    articles = articles.set_index("article_id")
    ordered = article_idx.sort_values("article_idx")["article_id"]
    texts = articles.loc[ordered]["text"].tolist()

    print(f"Encoding {len(texts):,} articles with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-norm → cosine = dot product
    )
    np.save(MODELS_DIR / "article_embeddings.npy", embeddings.astype(np.float32))
    print(f"Saved embeddings shape: {embeddings.shape}")
    return embeddings


def user_vector(customer_id: str, article_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute a user's taste vector as the purchase-count-weighted mean of
    the embeddings of articles they have bought.
    """
    counts = load_table(PROCESSED_DIR / "user_item_counts")
    article_idx = load_table(PROCESSED_DIR / "article_index")
    a_to_idx = dict(zip(article_idx["article_id"], article_idx["article_idx"]))

    user_counts = counts[counts["customer_id"] == customer_id]
    if user_counts.empty:
        raise ValueError(f"No purchase history for customer {customer_id}")

    vecs, weights = [], []
    for _, row in user_counts.iterrows():
        idx = a_to_idx.get(row["article_id"])
        if idx is not None:
            vecs.append(article_embeddings[idx])
            weights.append(row["count"])

    vecs = np.array(vecs)
    weights = np.array(weights, dtype=np.float32)
    uv = (vecs * weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(uv)
    return uv / norm if norm > 0 else uv


def cosine_scores(user_vec: np.ndarray, article_embeddings: np.ndarray) -> np.ndarray:
    """Return cosine similarity of user_vec against all article embeddings."""
    return article_embeddings @ user_vec  # embeddings already unit-normed


if __name__ == "__main__":
    build_embeddings()
