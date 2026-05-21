"""
render_bootstrap.py

Prepare a small real dataset and cache for Render deployments so the app is
usable without relying on the synthetic demo artifacts.
"""

import os
import subprocess
from pathlib import Path


def _run(cmd: list[str]):
    """Run a bootstrap subprocess and stream its command to stdout."""
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _exists(path: str) -> bool:
    return Path(path).exists()


def main():
    """Prepare a lightweight dataset, model, and cache for Render startup."""
    sample_rows = os.environ.get("RENDER_SAMPLE_TRANSACTIONS", "50000")
    cache_users = os.environ.get("RENDER_CACHE_USERS", "3")
    training_factors = os.environ.get("RENDER_TRAIN_FACTORS", "16")
    training_epochs = os.environ.get("RENDER_TRAIN_EPOCHS", "4")

    if not _exists("data/processed/customer_index.pkl"):
        _run(["python", "-m", "src.download_data", "--sample-transactions", sample_rows])
        _run(["python", "-m", "src.preprocessing"])

    if not _exists("models/als_model.pkl"):
        try:
            _run([
                "python",
                "-m",
                "src.training",
                "--factors",
                training_factors,
                "--epochs",
                training_epochs,
            ])
        except subprocess.CalledProcessError:
            print(
                "ALS training failed; continuing with content-only "
                "recommendation fallback.",
                flush=True,
            )

    if not _exists("models/article_embeddings.npy"):
        _run(["python", "-m", "src.content_based"])

    if not _exists("demo_cache/taste_profiles.json"):
        Path("demo_cache").mkdir(exist_ok=True)

    _run([
        "python",
        "-m",
        "src.inference",
        "--cache",
        "--cache-users",
        cache_users,
        "--top_k",
        "10",
    ])


if __name__ == "__main__":
    main()
