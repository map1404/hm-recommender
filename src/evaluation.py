"""
evaluation.py — Tianshi
Evaluates the ALS model on a held-out test split.

Metrics: Precision@K, Recall@K, NDCG@K
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
K = 10
TEST_FRAC = 0.2
RANDOM_STATE = 42


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate():
    print("Loading data...")
    user_item = load_npz(MODELS_DIR / "user_item_matrix.npz")
    with open(MODELS_DIR / "als_model.pkl", "rb") as f:
        model = pickle.load(f)

    counts = pd.read_parquet(PROCESSED_DIR / "user_item_counts.parquet")
    customer_idx = pd.read_parquet(PROCESSED_DIR / "customer_index.parquet")
    article_idx = pd.read_parquet(PROCESSED_DIR / "article_index.parquet")

    c_to_idx = dict(zip(customer_idx["customer_id"], customer_idx["customer_idx"]))
    idx_to_a = dict(zip(article_idx["article_idx"], article_idx["article_id"]))

    # Temporal split: hold out each user's most recent 20% of items
    transactions = pd.read_parquet(PROCESSED_DIR / "transactions.parquet")
    transactions = transactions.sort_values("t_dat")
    test_items: dict[str, set] = {}
    for cid, grp in transactions.groupby("customer_id"):
        n_test = max(1, int(len(grp) * TEST_FRAC))
        test_items[cid] = set(grp.tail(n_test)["article_id"])

    sample_users = list(test_items.keys())
    np.random.seed(RANDOM_STATE)
    sample_users = np.random.choice(sample_users, size=min(500, len(sample_users)), replace=False)

    precisions, recalls, ndcgs = [], [], []
    item_user = user_item.T.tocsr()

    for cid in sample_users:
        if cid not in c_to_idx:
            continue
        uid = c_to_idx[cid]
        recs, _ = model.recommend(uid, item_user[:, uid], N=K, filter_already_liked=True)
        recommended_articles = [idx_to_a.get(r) for r in recs if idx_to_a.get(r)]
        relevant = test_items[cid]
        hits = len(set(recommended_articles) & relevant)
        precisions.append(hits / K)
        recalls.append(hits / len(relevant) if relevant else 0)
        ndcgs.append(ndcg_at_k(recommended_articles, relevant, K))

    print(f"\n=== Evaluation results (K={K}, n={len(precisions)}) ===")
    print(f"  Precision@{K}: {np.mean(precisions):.4f}")
    print(f"  Recall@{K}:    {np.mean(recalls):.4f}")
    print(f"  NDCG@{K}:      {np.mean(ndcgs):.4f}")
    return {
        f"precision@{K}": np.mean(precisions),
        f"recall@{K}": np.mean(recalls),
        f"ndcg@{K}": np.mean(ndcgs),
    }


if __name__ == "__main__":
    evaluate()
