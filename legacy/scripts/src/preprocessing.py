"""
preprocessing.py
----------------
Data Engineer (Akash) module.

Cleans raw H&M CSVs and exports processed Parquet files.

Outputs:
  data/processed/transactions.parquet
  data/processed/articles.parquet
  data/processed/customers.parquet
  data/processed/user_item_counts.parquet
  data/processed/customer_index.parquet
  data/processed/article_index.parquet
  models/user_item_matrix.npz
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz

from src.storage import save_table

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MIN_PURCHASES = 5
LOOKBACK_DAYS = 180


def load_raw():
    print("Loading raw CSVs...")
    paths = {
        "transactions": RAW_DIR / "transactions_train.csv",
        "articles": RAW_DIR / "articles.csv",
        "customers": RAW_DIR / "customers.csv",
    }
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data files: {', '.join(missing)}. "
            "Run `python src/download_data.py` first."
        )
    transactions = pd.read_csv(paths["transactions"], dtype={"article_id": str})
    articles = pd.read_csv(paths["articles"], dtype={"article_id": str})
    customers = pd.read_csv(paths["customers"])
    return transactions, articles, customers


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    cutoff = df["t_dat"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
    df = df[df["t_dat"] >= cutoff].copy()
    counts = df.groupby("customer_id").size()
    active = counts[counts >= MIN_PURCHASES].index
    df = df[df["customer_id"].isin(active)].copy()
    print(f"  Transactions after filtering: {len(df):,}")
    print(f"  Unique customers: {df['customer_id'].nunique():,}")
    print(f"  Unique articles:  {df['article_id'].nunique():,}")
    return df


def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = [
        "product_type_name", "product_group_name", "graphical_appearance_name",
        "colour_group_name", "perceived_colour_value_name", "perceived_colour_master_name",
        "department_name", "index_name", "index_group_name",
        "section_name", "garment_group_name", "detail_desc",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df["text"] = df[[c for c in text_cols if c in df.columns]].agg(" | ".join, axis=1)
    return df


def build_user_item_matrix(transactions: pd.DataFrame):
    counts = (
        transactions.groupby(["customer_id", "article_id"])
        .size()
        .reset_index(name="count")
    )
    customer_ids = counts["customer_id"].unique()
    article_ids = counts["article_id"].unique()
    customer_idx = {c: i for i, c in enumerate(customer_ids)}
    article_idx = {a: i for i, a in enumerate(article_ids)}

    rows = counts["customer_id"].map(customer_idx)
    cols = counts["article_id"].map(article_idx)
    data = counts["count"].values

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(customer_ids), len(article_ids)),
    )

    save_table(
        pd.DataFrame({"customer_id": customer_ids, "customer_idx": range(len(customer_ids))}),
        PROCESSED_DIR / "customer_index",
    )
    save_table(
        pd.DataFrame({"article_id": article_ids, "article_idx": range(len(article_ids))}),
        PROCESSED_DIR / "article_index",
    )
    save_table(counts, PROCESSED_DIR / "user_item_counts")
    save_npz(MODELS_DIR / "user_item_matrix.npz", matrix)
    print(f"  User-item matrix shape: {matrix.shape}  nnz={matrix.nnz:,}")
    return matrix


def main():
    transactions, articles, customers = load_raw()
    transactions = clean_transactions(transactions)
    articles = clean_articles(articles)

    active_articles = transactions["article_id"].unique()
    articles = articles[articles["article_id"].isin(active_articles)].copy()

    save_table(transactions, PROCESSED_DIR / "transactions")
    save_table(articles, PROCESSED_DIR / "articles")
    save_table(customers, PROCESSED_DIR / "customers")
    print("Saved transactions, articles, customers.")
    build_user_item_matrix(transactions)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
