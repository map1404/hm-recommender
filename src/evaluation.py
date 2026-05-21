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
    user_item = load_npz(MODELS_DIR / "user_item_matrix.npz").tocsr()
    with open(MODELS_DIR / "als_model.pkl", "rb") as f:
        model = pickle.load(f)

    customer_idx = pd.read_parquet(PROCESSED_DIR / "customer_index.parquet")
    article_idx = pd.read_parquet(PROCESSED_DIR / "article_index.parquet")

    c_to_idx = dict(zip(customer_idx["customer_id"], customer_idx["customer_idx"]))
    idx_to_a = dict(zip(article_idx["article_idx"], article_idx["article_id"]))

    # training.py fits ALS on user_item.T (item × user), so implicit's internal
    # semantics are flipped: model.user_factors is indexed by article_idx and
    # model.item_factors is indexed by customer_idx.
    customer_factors = np.asarray(model.item_factors)   # (n_customers, factors)
    article_factors = np.asarray(model.user_factors)    # (n_articles, factors)
    n_articles = article_factors.shape[0]

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

    for cid in sample_users:
        if cid not in c_to_idx:
            continue
        uid = c_to_idx[cid]
        if uid >= customer_factors.shape[0]:
            continue
        scores = article_factors @ customer_factors[uid]
        purchased = user_item[uid].indices
        if len(purchased):
            scores[purchased] = -np.inf
        top_n = min(K, n_articles)
        top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        recommended_articles = [idx_to_a.get(int(r)) for r in top_idx if idx_to_a.get(int(r))]
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


def evaluate_popularity_baseline(k: int = K, recent_days: int = 60, sample_size: int = 500):
    """
    Popularity baseline using the train split only, evaluated against
    each user's held-out tail. Avoids leakage by computing popular items
    strictly from the pre-cutoff transactions.
    """
    print("Loading transactions for popularity baseline...")
    transactions = pd.read_parquet(PROCESSED_DIR / "transactions.parquet")
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions = transactions.sort_values("t_dat")

    train_rows, test_items = [], {}
    for cid, grp in transactions.groupby("customer_id"):
        n_test = max(1, int(len(grp) * TEST_FRAC))
        test_items[cid] = set(grp.tail(n_test)["article_id"])
        if len(grp) > n_test:
            train_rows.append(grp.head(len(grp) - n_test))
    train = pd.concat(train_rows, ignore_index=True) if train_rows else transactions.iloc[0:0]

    cutoff = train["t_dat"].max() - pd.Timedelta(days=recent_days)
    recent = train[train["t_dat"] >= cutoff]
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
    top_articles = (
        grouped.sort_values("popularity_score", ascending=False)
        .head(k)["article_id"]
        .tolist()
    )

    sample_users = list(test_items.keys())
    np.random.seed(RANDOM_STATE)
    sample_users = np.random.choice(
        sample_users, size=min(sample_size, len(sample_users)), replace=False
    )

    precisions, recalls, ndcgs = [], [], []
    for cid in sample_users:
        relevant = test_items[cid]
        hits = len(set(top_articles) & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant) if relevant else 0)
        ndcgs.append(ndcg_at_k(top_articles, relevant, k))

    print(f"\n=== Popularity baseline (K={k}, n={len(precisions)}) ===")
    print(f"  Precision@{k}: {np.mean(precisions):.4f}")
    print(f"  Recall@{k}:    {np.mean(recalls):.4f}")
    print(f"  NDCG@{k}:      {np.mean(ndcgs):.4f}")
    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
    }


if __name__ == "__main__":
    import sys
    if "--popularity" in sys.argv:
        evaluate_popularity_baseline()
    else:
        evaluate()
        try:
            evaluate_popularity_baseline()
        except FileNotFoundError as e:
            print(f"Skipping popularity baseline: {e}")
