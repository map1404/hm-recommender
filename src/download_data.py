"""
download_data.py

Download the H&M raw CSVs from a public Hugging Face mirror, avoiding the
Kaggle login flow. Supports downloading the full files or a smaller sampled
subset suitable for local development.
"""

import argparse
import csv
import itertools
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
BASE_URL = (
    "https://huggingface.co/datasets/"
    "einrafh/hnm-fashion-recommendations-data/resolve/main/data/raw"
)
FILES = {
    "articles": ("articles.csv", "article_id"),
    "customers": ("customers.csv", "customer_id"),
    "transactions": ("transactions_train.csv", None),
}
CHUNK_SIZE = 1024 * 1024
TIMEOUT_SECONDS = 300


def _stream(url: str):
    response = requests.get(url, stream=True, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response


def _download_file(filename: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / filename
    url = f"{BASE_URL}/{filename}?download=true"

    print(f"Downloading {filename}...")
    with _stream(url) as response, destination.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                fh.write(chunk)
    print(f"  Saved {destination}")


def _sample_transactions(sample_rows: int):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / "transactions_train.csv"
    url = f"{BASE_URL}/transactions_train.csv?download=true"
    customer_ids = set()
    article_ids = set()

    print(f"Downloading first {sample_rows:,} transaction rows...")
    with _stream(url) as response, destination.open("w", newline="") as fh:
        lines = response.iter_lines(decode_unicode=True)
        reader = csv.DictReader(lines)
        writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in itertools.islice(reader, sample_rows):
            writer.writerow(row)
            customer_ids.add(row["customer_id"])
            article_ids.add(str(row["article_id"]))

    print(f"  Saved {destination}")
    return customer_ids, article_ids


def _filter_dimension(filename: str, key: str, allowed_values: set[str]):
    destination = RAW_DIR / filename
    url = f"{BASE_URL}/{filename}?download=true"

    print(f"Filtering {filename} for sampled transactions...")
    with _stream(url) as response, destination.open("w", newline="") as fh:
        lines = response.iter_lines(decode_unicode=True)
        reader = csv.DictReader(lines)
        writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
        writer.writeheader()

        kept = 0
        for row in reader:
            if str(row[key]) in allowed_values:
                writer.writerow(row)
                kept += 1

    print(f"  Saved {destination} ({kept:,} rows)")


def main():
    """Download the full dataset or a reduced local-development sample."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-transactions",
        type=int,
        default=None,
        help=(
            "Download only the first N transaction rows and filter customers/articles "
            "to matching IDs for a smaller local-dev dataset."
        ),
    )
    args = parser.parse_args()

    if args.sample_transactions is not None:
        if args.sample_transactions <= 0:
            raise ValueError("--sample-transactions must be a positive integer")
        customer_ids, article_ids = _sample_transactions(args.sample_transactions)
        _filter_dimension("customers.csv", "customer_id", customer_ids)
        _filter_dimension("articles.csv", "article_id", article_ids)
        return

    for filename, _ in FILES.values():
        _download_file(filename)


if __name__ == "__main__":
    main()
