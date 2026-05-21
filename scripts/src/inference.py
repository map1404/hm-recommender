"""
inference.py
------------
ML Developer B (Fuwei) module.

End-to-end pipeline: customer_id → taste profile + top-10 recs + explanations.

Usage:
  # Run live for one customer:
  python src/inference.py --customer_id <id>

  # Pre-generate cache for all curated demo IDs:
  python src/inference.py --cache

  # Force live LLM even if cached:
  python src/inference.py --customer_id <id> --live
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.hybrid import recommend
from src.llm import generate_taste_profile, explain_recommendation
from src.storage import load_table

DEMO_CACHE_DIR = Path("demo_cache")
DEMO_CACHE_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = Path("data/processed")

TASTE_PROFILES_PATH = DEMO_CACHE_DIR / "taste_profiles.json"
EXPLANATIONS_PATH = DEMO_CACHE_DIR / "explanations.json"
RECOMMENDATIONS_PATH = DEMO_CACHE_DIR / "recommendations.json"


def _load_cache():
    profiles = json.loads(TASTE_PROFILES_PATH.read_text()) if TASTE_PROFILES_PATH.exists() else {}
    explanations = json.loads(EXPLANATIONS_PATH.read_text()) if EXPLANATIONS_PATH.exists() else {}
    recs = json.loads(RECOMMENDATIONS_PATH.read_text()) if RECOMMENDATIONS_PATH.exists() else {}
    return profiles, explanations, recs


def get_demo_customer_ids(limit: int = 25) -> list[str]:
    """Pick customers with most purchases from the processed data."""
    curated_path = DEMO_CACHE_DIR / "curated_ids.json"
    if curated_path.exists():
        return json.loads(curated_path.read_text())

    transactions = load_table(PROCESSED_DIR / "transactions")
    top = (
        transactions.groupby("customer_id")["article_id"]
        .count()
        .sort_values(ascending=False)
        .head(limit)
        .index.tolist()
    )
    with open(curated_path, "w") as f:
        json.dump(top, f, indent=2)
    return top


def run_live(customer_id: str, top_k: int = 10) -> dict:
    """Full real-time pipeline for one customer."""
    print(f"Generating taste profile for {customer_id[:16]}...")
    taste_profile = generate_taste_profile(customer_id)

    print("Running hybrid recommender...")
    articles = recommend(customer_id, top_k=top_k)

    print("Generating per-item explanations...")
    for article in articles:
        article["explanation"] = explain_recommendation(taste_profile, article)

    return {
        "customer_id": customer_id,
        "taste_profile": taste_profile,
        "recommendations": articles,
    }


def run_cached(customer_id: str, top_k: int = 10) -> dict:
    """Load pre-generated outputs from demo_cache/."""
    profiles, explanations, cached_recs = _load_cache()
    if customer_id not in profiles:
        raise KeyError(
            f"Customer {customer_id[:16]} not in cache. "
            "Run `python src/inference.py --cache` or use --live flag."
        )
    articles = cached_recs.get(customer_id) or recommend(customer_id, top_k=top_k)
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


def build_cache(customer_ids: list[str] = None, top_k: int = 10):
    """Pre-generate and persist taste profiles + explanations for all demo IDs."""
    if customer_ids is None:
        customer_ids = get_demo_customer_ids()

    profiles, explanations, all_recs = _load_cache()

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
                exp = explain_recommendation(taste_profile, article)
                cid_explanations[article["article_id"]] = exp
                article["explanation"] = exp
            profiles[cid] = taste_profile
            explanations[cid] = cid_explanations
            all_recs[cid] = [{k: v for k, v in a.items() if k != "explanation"} for a in articles]
        except Exception as e:
            print(f"  ERROR for {cid[:16]}: {e}")

    TASTE_PROFILES_PATH.write_text(json.dumps(profiles, indent=2))
    EXPLANATIONS_PATH.write_text(json.dumps(explanations, indent=2))
    RECOMMENDATIONS_PATH.write_text(json.dumps(all_recs, indent=2))
    print(f"Cache written → {DEMO_CACHE_DIR}/ ({len(profiles)} customers)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer_id", type=str, default=None)
    parser.add_argument("--cache", action="store_true", help="Pre-generate cache for all demo IDs")
    parser.add_argument("--live", action="store_true", help="Force real-time LLM call")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    if args.cache:
        build_cache(top_k=args.top_k)
    elif args.customer_id:
        fn = run_live if args.live else run_cached
        result = fn(args.customer_id, top_k=args.top_k)
        print(f"\nTaste profile:\n{result['taste_profile']}\n")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"{i}. {rec['prod_name']} — {rec.get('explanation', '')}")
    else:
        parser.print_help()
