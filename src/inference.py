"""
inference.py — Fuwei
End-to-end pipeline: customer_id → taste profile + top-10 recommendations + explanations.

Usage:
  # Run live for one customer:
  python src/inference.py --customer_id 000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318

  # Pre-generate cache for all curated demo IDs:
  python src/inference.py --cache
"""

import argparse
import json
from pathlib import Path

from src.hybrid import recommend
from src.llm import generate_taste_profile, generate_explanation
from src.popularity import is_cold_start_customer, recommend_popular
from src.storage import load_table

DEMO_CACHE_DIR = Path("demo_cache")
DEMO_CACHE_DIR.mkdir(exist_ok=True)

TASTE_PROFILES_PATH = DEMO_CACHE_DIR / "taste_profiles.json"
EXPLANATIONS_PATH = DEMO_CACHE_DIR / "explanations.json"

# ── Curated demo customer IDs (replace with real IDs from your dataset) ──────
DEFAULT_DEMO_CUSTOMER_IDS = [
    "000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318",
    "0000c152dc000ad04a760d5ec7d39cfc6fcb46fb4791c5ffe7e001fd9fdc0219",
    "00013dbbc5e89286a6b8b7b5f7f8ded69e60b2e7f4b0b2e6e4ef7b6f2d6a6e4",
    # Add 20-27 more IDs covering different style archetypes
]


def demo_customer_ids(limit: int = 3) -> list[str]:
    customer_index_path = Path("data/processed/customer_index.pkl")
    if customer_index_path.exists():
        customer_index = load_table(Path("data/processed/customer_index"))
        return customer_index["customer_id"].head(limit).tolist()
    return DEFAULT_DEMO_CUSTOMER_IDS[:limit]


def run_live(customer_id: str, top_k: int = 10) -> dict:
    """Full real-time pipeline for one customer."""
    if is_cold_start_customer(customer_id):
        print("Cold-start customer — using popularity fallback.")
        return {
            "customer_id": customer_id,
            "taste_profile": (
                "Cold-start mode: this user has no purchase history, so the "
                "system recommends recently popular H&M products."
            ),
            "recommendations": recommend_popular(top_k=top_k),
        }

    print(f"Generating taste profile for {customer_id[:16]}...")
    taste_profile = generate_taste_profile(customer_id)

    print("Running hybrid recommender...")
    articles = recommend(customer_id, top_k=top_k)

    print("Generating per-item explanations...")
    for article in articles:
        article["explanation"] = generate_explanation(customer_id, taste_profile, article)

    return {
        "customer_id": customer_id,
        "taste_profile": taste_profile,
        "recommendations": articles,
    }


def load_cache() -> tuple[dict, dict]:
    profiles = json.loads(TASTE_PROFILES_PATH.read_text()) if TASTE_PROFILES_PATH.exists() else {}
    explanations = json.loads(EXPLANATIONS_PATH.read_text()) if EXPLANATIONS_PATH.exists() else {}
    return profiles, explanations


def run_cached(customer_id: str, top_k: int = 10) -> dict:
    """Load pre-generated outputs from demo_cache/."""
    if is_cold_start_customer(customer_id):
        return {
            "customer_id": customer_id,
            "taste_profile": (
                "Cold-start mode: this user has no purchase history, so the "
                "system recommends recently popular H&M products."
            ),
            "recommendations": recommend_popular(top_k=top_k),
        }

    profiles, explanations = load_cache()
    if customer_id not in profiles:
        raise KeyError(
            f"Customer {customer_id[:16]} not in cache. "
            "Run with --cache to pre-generate, or use --live flag."
        )
    articles = recommend(customer_id, top_k=top_k)
    taste_profile = profiles[customer_id]
    cust_explanations = explanations.get(customer_id, {})
    for article in articles:
        article["explanation"] = cust_explanations.get(
            article["article_id"], "No cached explanation."
        )
    return {
        "customer_id": customer_id,
        "taste_profile": taste_profile,
        "recommendations": articles,
    }


def build_cache(customer_ids: list[str] | None = None, top_k: int = 10):
    """Pre-generate and persist taste profiles + explanations for all demo IDs."""
    if customer_ids is None:
        customer_ids = demo_customer_ids()

    profiles, explanations = load_cache()

    for cid in customer_ids:
        if cid in profiles:
            print(f"  Skipping {cid[:16]} (already cached)")
            continue
        print(f"  Caching {cid[:16]}...")
        try:
            taste_profile = generate_taste_profile(cid)
            articles = recommend(cid, top_k=top_k)
            cid_explanations = {}
            for article in articles:
                exp = generate_explanation(cid, taste_profile, article)
                cid_explanations[article["article_id"]] = exp
                article["explanation"] = exp
            profiles[cid] = taste_profile
            explanations[cid] = cid_explanations
        except Exception as e:
            print(f"  ERROR for {cid[:16]}: {e}")

    TASTE_PROFILES_PATH.write_text(json.dumps(profiles, indent=2))
    EXPLANATIONS_PATH.write_text(json.dumps(explanations, indent=2))
    print(f"Cache written → {DEMO_CACHE_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer_id", type=str, default=None)
    parser.add_argument("--cache", action="store_true", help="Pre-generate cache for all demo IDs")
    parser.add_argument("--live", action="store_true", help="Force real-time LLM call")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--cache-users", type=int, default=3)
    args = parser.parse_args()

    if args.cache:
        build_cache(customer_ids=demo_customer_ids(args.cache_users), top_k=args.top_k)
    elif args.customer_id:
        fn = run_live if args.live else run_cached
        result = fn(args.customer_id, top_k=args.top_k)
        print(f"\nTaste profile:\n{result['taste_profile']}\n")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"{i}. {rec['prod_name']} — {rec['explanation']}")
    else:
        parser.print_help()
