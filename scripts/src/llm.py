"""
llm.py
------
ML Developer B (Fuwei) module.

Two LLM-powered functions:
  1. generate_taste_profile(customer_id) → str
  2. explain_recommendation(taste_profile, article) → str

Uses OpenAI API (gpt-4o-mini). 
Falls back to rule-based summaries if the API is unavailable.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from src.storage import load_table

load_dotenv()

logger = logging.getLogger(__name__)
PROCESSED_DIR = Path("data/processed")


def _get_client():
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        return OpenAI(api_key=api_key)
    except Exception as e:
        logger.warning(f"OpenAI client unavailable: {e}")
        return None


def _purchase_summary(customer_id: str, n: int = 30) -> str:
    transactions = load_table(PROCESSED_DIR / "transactions")
    articles = load_table(PROCESSED_DIR / "articles").set_index("article_id")

    history = (
        transactions[transactions["customer_id"] == customer_id]
        .sort_values("t_dat", ascending=False)
        .head(n)
    )
    if history.empty:
        return "No purchase history available."

    lines = []
    for _, row in history.iterrows():
        aid = row["article_id"]
        if aid in articles.index:
            a = articles.loc[aid]
            lines.append(
                f"- {a.get('prod_name', aid)} "
                f"({a.get('product_type_name', '')}, "
                f"{a.get('colour_group_name', '')}, "
                f"{a.get('garment_group_name', '')})"
            )
    return "\n".join(lines) if lines else "No recognisable items."


def generate_taste_profile(customer_id: str, max_tokens: int = 300) -> str:
    summary = _purchase_summary(customer_id)

    prompt = f"""You are a personal stylist analysing a shopper's purchase history.

Here are their recent H&M purchases:
{summary}

Write a concise 2-3 sentence taste profile that captures:
- Their preferred style category (casual, formal, sporty, etc.)
- Dominant colours or colour palette
- Preferred garment types or silhouettes

Write in second person ("You prefer..."). Be specific, not generic.
Output only the taste profile text, no preamble."""

    client = _get_client()
    if client:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI taste profile failed: {e}; using fallback")

    return _fallback_taste_profile(customer_id)


def explain_recommendation(
    taste_profile: str,
    article: dict,
    max_tokens: int = 128,
) -> str:
    prompt = f"""You are a personal stylist at H&M.

Customer taste profile: {taste_profile}

Recommended article:
- Name: {article.get('prod_name', '')}
- Type: {article.get('product_type_name', '')}
- Colour: {article.get('colour_group_name', '')}
- Garment group: {article.get('garment_group_name', '')}

Write exactly one sentence explaining why this item suits this customer.
Reference at least one specific attribute that links the item to their taste profile.
Do not use generic phrases like "perfect for you" or "you'll love this"."""

    client = _get_client()
    if client:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI explanation failed: {e}; using fallback")

    return _fallback_explanation(article)


def _fallback_taste_profile(customer_id: str) -> str:
    try:
        from collections import Counter
        transactions = load_table(PROCESSED_DIR / "transactions")
        articles = load_table(PROCESSED_DIR / "articles").set_index("article_id")
        history = transactions[transactions["customer_id"] == customer_id]
        types = Counter(
            articles.loc[aid].get("product_type_name", "")
            for aid in history["article_id"]
            if aid in articles.index
        )
        colours = Counter(
            articles.loc[aid].get("colour_group_name", "")
            for aid in history["article_id"]
            if aid in articles.index
        )
        top_types = [t for t, _ in types.most_common(3) if t]
        top_colours = [c for c, _ in colours.most_common(3) if c]
        parts = []
        if top_types:
            parts.append(f"Your wardrobe centres on {', '.join(top_types).lower()}")
        if top_colours:
            parts.append(f"with a preference for {', '.join(top_colours).lower()} tones")
        return ". ".join(parts) + "." if parts else "Your style profile is still developing."
    except Exception:
        return "Your style profile is still developing."


def _fallback_explanation(article: dict) -> str:
    name = article.get("prod_name", "This item")
    colour = article.get("colour_group_name", "")
    type_ = article.get("product_type_name", "")
    colour_part = f" in {colour.lower()}" if colour else ""
    type_part = f" — a {type_.lower()}" if type_ else ""
    return f"{name}{colour_part}{type_part} aligns with the style patterns in your purchase history."
