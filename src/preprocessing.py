"""
preprocessing.py — Akash
Cleans raw H&M CSVs and exports processed Parquet files.

Outputs:
  data/processed/transactions.parquet   — filtered transactions
  data/processed/articles.parquet       — cleaned article metadata
  data/processed/customers.parquet      — cleaned customer metadata
  data/processed/user_item_counts.parquet — user × article purchase counts
"""

from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from src.storage import save_table

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MIN_PURCHASES = 5          # minimum transactions per customer to keep
LOOKBACK_DAYS = 180        # last N days of transactions to use


def load_raw():
    """Load the raw H&M transaction, article, and customer CSV files."""
    print("Loading raw CSVs...")
    transactions_path = RAW_DIR / "transactions_train.csv"
    articles_path = RAW_DIR / "articles.csv"
    customers_path = RAW_DIR / "customers.csv"
    missing = [
        path.name
        for path in (transactions_path, articles_path, customers_path)
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing raw data files: {missing_text}. "
            "Run `python src/download_data.py` first."
        )

    transactions = pd.read_csv(transactions_path, dtype={"article_id": str})
    articles = pd.read_csv(articles_path, dtype={"article_id": str})
    customers = pd.read_csv(customers_path)
    return transactions, articles, customers


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Filter transactions to the recent active-customer subset."""
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    cutoff = df["t_dat"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
    df = df[df["t_dat"] >= cutoff].copy()
    # Keep customers with enough history
    counts = df.groupby("customer_id").size()
    active = counts[counts >= MIN_PURCHASES].index
    df = df[df["customer_id"].isin(active)].copy()
    print(f"  Transactions after filtering: {len(df):,}")
    print(f"  Unique customers: {df['customer_id'].nunique():,}")
    print(f"  Unique articles:  {df['article_id'].nunique():,}")
    return df


def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Create a text field used by the content-based embedding model."""
    # Build a rich text description for embedding
    text_cols = [
        "product_type_name", "product_group_name", "graphical_appearance_name",
        "colour_group_name", "perceived_colour_value_name", "perceived_colour_master_name",
        "department_name", "index_name", "index_group_name", "section_name", "garment_group_name",
        "detail_desc",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)
    df["text"] = df[text_cols].agg(" | ".join, axis=1)
    return df


def build_user_item_matrix(transactions: pd.DataFrame):
    """Returns a sparse CSR matrix of purchase counts plus index mappings."""
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

    # Save index mappings alongside matrix for later retrieval
    idx_df = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_idx": range(len(customer_ids)),
    })
    art_df = pd.DataFrame({
        "article_id": article_ids,
        "article_idx": range(len(article_ids)),
    })
    save_table(idx_df, PROCESSED_DIR / "customer_index")
    save_table(art_df, PROCESSED_DIR / "article_index")
    save_table(counts, PROCESSED_DIR / "user_item_counts")
    save_npz(MODELS_DIR / "user_item_matrix.npz", matrix)
    print(f"  User-item matrix shape: {matrix.shape}  nnz={matrix.nnz:,}")
    return matrix, customer_idx, article_idx


def main():
    """Run the preprocessing pipeline and persist processed artifacts."""
    transactions, articles, customers = load_raw()

    transactions = clean_transactions(transactions)
    articles = clean_articles(articles)

    # Keep only articles present in filtered transactions
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
