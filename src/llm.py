"""
llm.py — Fuwei
Generates:
  1. A natural-language taste profile summary for a customer.
  2. A one-sentence personalised explanation for each recommended item.

Uses the OpenAI Responses API.
Set OPENAI_API_KEY in your .env file.
"""

import os
import time
from pathlib import Path

import requests
from requests import HTTPError
from dotenv import load_dotenv
from src.storage import load_table

load_dotenv()

PROCESSED_DIR = Path("data/processed")
API_URL = "https://api.openai.com/v1/responses"
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
MAX_TOKENS = 512
REQUEST_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "90"))
MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "6"))
RETRY_BASE_SECONDS = float(os.environ.get("OPENAI_RETRY_BASE_SECONDS", "10"))


def _api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to your .env file.")
    return api_key


def _fallback_taste_profile(summary: str) -> str:
    lines = [line for line in summary.splitlines() if line.strip()]
    if not lines:
        return (
            "You prefer practical everyday basics with a consistent wardrobe pattern. "
            "Your purchase history is limited, so this profile is a lightweight fallback."
        )

    joined = " ".join(lines[:8]).lower()
    style = "casual"
    if any(token in joined for token in ["dress", "blouse", "skirt", "satin"]):
        style = "feminine and polished"
    elif any(token in joined for token in ["hoodie", "leggings", "running", "technical"]):
        style = "sporty and functional"
    elif any(token in joined for token in ["blazer", "trousers", "shirt", "tailored"]):
        style = "minimal and polished"

    colours = []
    for colour in ["black", "white", "beige", "blue", "grey", "green", "pink", "red", "cream"]:
        if colour in joined:
            colours.append(colour)
    colour_text = ", ".join(colours[:3]) if colours else "neutral tones"

    return (
        f"You prefer a {style} wardrobe built around {colour_text}. "
        "Your purchase history leans toward repeatable everyday silhouettes and familiar categories."
    )


def _fallback_explanation(_taste_profile: str, article: dict) -> str:
    colour = article.get("colour_group_name", "overall")
    product_type = article.get("product_type_name", "piece").lower()
    garment_group = article.get("garment_group_name", "wardrobe").lower()
    return (
        f"This {product_type} matches your profile through its {colour.lower()} palette "
        f"and {garment_group} styling cues."
    )


def _generate_text(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    response = None
    for attempt in range(MAX_RETRIES):
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "max_output_tokens": max_tokens,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "store": False,
            },
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 429:
            break

        if attempt == MAX_RETRIES - 1:
            break

        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            sleep_seconds = max(float(retry_after), RETRY_BASE_SECONDS)
        else:
            sleep_seconds = RETRY_BASE_SECONDS * (attempt + 1)
        time.sleep(sleep_seconds)

    try:
        response.raise_for_status()
    except HTTPError as exc:
        detail = response.text.strip()
        if detail:
            raise HTTPError(f"{exc}. Response body: {detail}") from exc
        raise
    payload = response.json()
    if payload.get("output_text"):
        return payload["output_text"].strip()

    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "").strip()

    raise ValueError(f"Unexpected OpenAI response payload: {payload}")


def _purchase_summary(customer_id: str, n: int = 30) -> str:
    """Build a short text summary of a customer's recent purchases."""
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


def generate_taste_profile(customer_id: str) -> str:
    """
    Returns a 2-3 sentence taste profile such as:
    'Prefers casual sportswear in neutral tones. Often buys fitted silhouettes
    and basics. Gravitates toward jersey fabrics and minimal prints.'
    """
    summary = _purchase_summary(customer_id)
    system_prompt = "You are a personal stylist analysing a shopper's purchase history."
    user_prompt = f"""Here are their recent H&M purchases:
{summary}

Write a concise 2-3 sentence taste profile that captures:
- Their preferred style category (casual, formal, sporty, etc.)
- Dominant colours or colour palette
- Preferred garment types or silhouettes
- Any notable patterns (prints, basics, occasion-driven)

Write in second person ("You prefer..."). Be specific, not generic.
Do not invent attributes not evidenced by the purchase list."""

    try:
        return _generate_text(system_prompt, user_prompt, MAX_TOKENS)
    except (EnvironmentError, HTTPError, requests.RequestException, ValueError):
        return _fallback_taste_profile(summary)


def generate_explanation(
    _customer_id: str,
    taste_profile: str,
    article: dict,
) -> str:
    """
    Returns one sentence explaining why this article suits this customer,
    referencing at least one concrete attribute from their history.
    """
    system_prompt = "You are a personal stylist at H&M."
    user_prompt = f"""Customer taste profile: {taste_profile}

Recommended article:
- Name: {article.get('prod_name', '')}
- Type: {article.get('product_type_name', '')}
- Colour: {article.get('colour_group_name', '')}
- Garment group: {article.get('garment_group_name', '')}

Write exactly one sentence explaining why this item suits this customer.
Reference at least one specific attribute
(colour, garment type, silhouette, or style)
that links the item back to their taste profile.
Do not use generic phrases like "perfect for you" or "you'll love this"."""

    try:
        return _generate_text(system_prompt, user_prompt, 128)
    except (EnvironmentError, HTTPError, requests.RequestException, ValueError):
        return _fallback_explanation(taste_profile, article)
