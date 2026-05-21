"""
generate_demo_cache.py
----------------------
Standalone script to populate demo_cache/ with synthetic data when
the full H&M dataset is not yet downloaded.

Usage:
  python scripts/generate_demo_cache.py --synthetic   # generate synthetic demo data
  python scripts/generate_demo_cache.py --live        # use real data + live LLM
"""

import json
import random
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEMO_CACHE_DIR = ROOT / "demo_cache"
DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic personas for demo (no dataset needed)
# ---------------------------------------------------------------------------

SYNTHETIC_CUSTOMERS = [
    {
        "customer_id": "DEMO_CASUAL_001",
        "archetype": "Casual / Sportswear",
        "taste_profile": (
            "Your wardrobe is built around relaxed, everyday comfort — you gravitate towards jersey tops, "
            "jogger trousers, and oversized hoodies in a palette of soft greys, whites, and muted navy. "
            "Fitted yet flexible silhouettes dominate your choices, and you rarely stray into formal territory. "
            "Ease of wear and versatility across seasons define your shopping behaviour."
        ),
        "items": [
            {"article_id": "0791587001", "prod_name": "Regular Fit Jersey Top", "product_type_name": "T-shirt",
             "colour_group_name": "White", "garment_group_name": "Jersey Basic", "price": 9.99,
             "explanation": "This relaxed white jersey top directly mirrors your preference for easy, neutral-toned everyday wear."},
            {"article_id": "0860345002", "prod_name": "Slim Jogger Trousers", "product_type_name": "Trousers",
             "colour_group_name": "Grey", "garment_group_name": "Trousers", "price": 24.99,
             "explanation": "The slim-fit grey jogger matches both your favourite silhouette and your go-to neutral palette."},
            {"article_id": "0562245001", "prod_name": "Oversized Hoodie", "product_type_name": "Hoodie",
             "colour_group_name": "Dark Grey", "garment_group_name": "Jersey Fancy", "price": 29.99,
             "explanation": "An oversized hoodie in dark grey aligns perfectly with your comfort-first, muted-tone aesthetic."},
            {"article_id": "0706016001", "prod_name": "5-Pack Cotton Socks", "product_type_name": "Socks",
             "colour_group_name": "White", "garment_group_name": "Socks and Tights", "price": 6.99,
             "explanation": "A practical white sock multipack supports your no-fuss, everyday wardrobe philosophy."},
            {"article_id": "0714790001", "prod_name": "Regular Fit Crewneck Sweatshirt", "product_type_name": "Sweatshirt",
             "colour_group_name": "Light Grey", "garment_group_name": "Jersey Basic", "price": 19.99,
             "explanation": "The relaxed crew neck silhouette in light grey fits seamlessly into your casual, neutral-toned rotation."},
            {"article_id": "0791587002", "prod_name": "Slim-Fit Jersey Shorts", "product_type_name": "Shorts",
             "colour_group_name": "Navy Blue", "garment_group_name": "Shorts", "price": 14.99,
             "explanation": "Navy slim-fit shorts bridge your sportswear habits and your preference for fitted silhouettes."},
            {"article_id": "0820230003", "prod_name": "Zip-through Fleece Jacket", "product_type_name": "Jacket",
             "colour_group_name": "Grey", "garment_group_name": "Outdoor", "price": 34.99,
             "explanation": "A grey fleece zip-through offers the layering versatility your casual, active lifestyle calls for."},
            {"article_id": "0706016002", "prod_name": "Long-sleeve Jersey Top", "product_type_name": "T-shirt",
             "colour_group_name": "White", "garment_group_name": "Jersey Basic", "price": 12.99,
             "explanation": "A clean white long-sleeve jersey top extends your wardrobe into cooler months without leaving your comfort zone."},
            {"article_id": "0562245002", "prod_name": "Straight-leg Sweatpants", "product_type_name": "Trousers",
             "colour_group_name": "Black", "garment_group_name": "Trousers", "price": 22.99,
             "explanation": "Black straight-leg sweatpants complement your existing grey-and-white casual essentials perfectly."},
            {"article_id": "0820230004", "prod_name": "Sporty Running Cap", "product_type_name": "Cap/Hat",
             "colour_group_name": "Grey", "garment_group_name": "Accessories", "price": 9.99,
             "explanation": "A minimal grey sports cap ties your active, casual aesthetic together with a practical finishing touch."},
        ],
    },
    {
        "customer_id": "DEMO_FORMAL_002",
        "archetype": "Formal / Work",
        "taste_profile": (
            "Your style centres on polished, work-ready pieces — you have a strong affinity for tailored blazers, "
            "slim-fit dress shirts, and structured trousers in a refined palette of navy, black, and cream. "
            "Clean lines and precise fits are consistent across your purchases, with occasional smart-casual pieces "
            "for weekend wear. Quality and longevity clearly take precedence over trend-chasing."
        ),
        "items": [
            {"article_id": "0610776001", "prod_name": "Slim-fit Blazer", "product_type_name": "Blazer",
             "colour_group_name": "Dark Blue", "garment_group_name": "Suits & Tailoring", "price": 59.99,
             "explanation": "A dark navy slim blazer is the exact kind of refined, tailored piece your work wardrobe is built around."},
            {"article_id": "0714790003", "prod_name": "Oxford Dress Shirt", "product_type_name": "Dress shirt",
             "colour_group_name": "White", "garment_group_name": "Shirts", "price": 29.99,
             "explanation": "A crisp white Oxford shirt anchors your preference for clean, office-ready essentials."},
            {"article_id": "0610776002", "prod_name": "Slim Suit Trousers", "product_type_name": "Trousers",
             "colour_group_name": "Black", "garment_group_name": "Suits & Tailoring", "price": 39.99,
             "explanation": "Black slim-cut trousers pair effortlessly with your tailored blazers and dress shirts."},
            {"article_id": "0706016003", "prod_name": "Striped Dress Shirt", "product_type_name": "Dress shirt",
             "colour_group_name": "Light Blue", "garment_group_name": "Shirts", "price": 27.99,
             "explanation": "A subtle light-blue stripe shirt adds quiet variety while staying firmly within your polished register."},
            {"article_id": "0820230005", "prod_name": "Wool-blend Overcoat", "product_type_name": "Coat",
             "colour_group_name": "Dark Grey", "garment_group_name": "Coats", "price": 89.99,
             "explanation": "A structured dark-grey overcoat extends your tailored aesthetic into outerwear with the same refined tone."},
            {"article_id": "0791587003", "prod_name": "Chino Slim Trousers", "product_type_name": "Trousers",
             "colour_group_name": "Beige", "garment_group_name": "Trousers", "price": 34.99,
             "explanation": "Beige slim chinos bridge your office precision with smart-casual versatility for weekend wear."},
            {"article_id": "0562245003", "prod_name": "Turtleneck Knitwear", "product_type_name": "Jumper/Knitwear",
             "colour_group_name": "Black", "garment_group_name": "Knitwear", "price": 34.99,
             "explanation": "A black turtleneck brings understated sophistication to your rotation, pairing beautifully with your blazers."},
            {"article_id": "0860345003", "prod_name": "Classic Leather Belt", "product_type_name": "Belt",
             "colour_group_name": "Dark Brown", "garment_group_name": "Accessories", "price": 19.99,
             "explanation": "A dark-brown leather belt completes the polished, well-considered look your formal wardrobe demands."},
            {"article_id": "0714790004", "prod_name": "Fine-knit V-neck Jumper", "product_type_name": "Jumper/Knitwear",
             "colour_group_name": "Navy Blue", "garment_group_name": "Knitwear", "price": 29.99,
             "explanation": "A fine-knit navy V-neck works seamlessly over your dress shirts as a smart layering piece."},
            {"article_id": "0706016004", "prod_name": "Regular-fit Linen Shirt", "product_type_name": "Shirt",
             "colour_group_name": "Light Beige", "garment_group_name": "Shirts", "price": 24.99,
             "explanation": "A light-beige linen shirt introduces subtle texture while maintaining the clean, elevated tone of your wardrobe."},
        ],
    },
    {
        "customer_id": "DEMO_FEMININE_003",
        "archetype": "Feminine / Occasion",
        "taste_profile": (
            "Your wardrobe is defined by feminine silhouettes and occasion-ready pieces — wrap dresses, "
            "floral blouses, and A-line skirts in a palette of soft pinks, florals, and warm creams dominate your history. "
            "You balance everyday femininity with a few statement pieces for special occasions, and your colour "
            "choices consistently lean warm and romantic."
        ),
        "items": [
            {"article_id": "0610776003", "prod_name": "Wrap Midi Dress", "product_type_name": "Dress",
             "colour_group_name": "Light Pink", "garment_group_name": "Dresses Ladies", "price": 39.99,
             "explanation": "A blush pink wrap midi dress is the archetypal piece in your feminine, occasion-ready wardrobe."},
            {"article_id": "0714790005", "prod_name": "Floral Chiffon Blouse", "product_type_name": "Blouse",
             "colour_group_name": "Pink/Floral", "garment_group_name": "Blouses", "price": 24.99,
             "explanation": "A floral chiffon blouse channels your consistent preference for romantic prints and warm tones."},
            {"article_id": "0791587004", "prod_name": "A-line Midi Skirt", "product_type_name": "Skirt",
             "colour_group_name": "Cream", "garment_group_name": "Skirts", "price": 29.99,
             "explanation": "A cream A-line midi skirt pairs effortlessly with your feminine blouses and complements your warm palette."},
            {"article_id": "0562245004", "prod_name": "Tie-front Smock Dress", "product_type_name": "Dress",
             "colour_group_name": "White", "garment_group_name": "Dresses Ladies", "price": 34.99,
             "explanation": "A white smock dress with tie-front detail adds effortless femininity and warmth-season versatility."},
            {"article_id": "0820230006", "prod_name": "Lace-trim Cami Top", "product_type_name": "Top",
             "colour_group_name": "Dusty Pink", "garment_group_name": "Tops", "price": 17.99,
             "explanation": "A dusty-pink lace cami layers beautifully over your dresses and mirrors your soft, romantic colour choices."},
            {"article_id": "0860345004", "prod_name": "Pleated Midi Skirt", "product_type_name": "Skirt",
             "colour_group_name": "Beige", "garment_group_name": "Skirts", "price": 27.99,
             "explanation": "A warm beige pleated midi skirt extends your love of flowy, feminine silhouettes into transitional dressing."},
            {"article_id": "0706016005", "prod_name": "Floral Wrap Blouse", "product_type_name": "Blouse",
             "colour_group_name": "Pink Flowers", "garment_group_name": "Blouses", "price": 22.99,
             "explanation": "Another floral print blouse in your signature warm-pink register — exactly the kind of piece you reach for most."},
            {"article_id": "0714790006", "prod_name": "Wide-leg Linen Trousers", "product_type_name": "Trousers",
             "colour_group_name": "White", "garment_group_name": "Trousers", "price": 32.99,
             "explanation": "White wide-leg linen trousers bring an airy, warm-weather elegance that aligns with your occasion-ready aesthetic."},
            {"article_id": "0610776004", "prod_name": "Off-shoulder Mini Dress", "product_type_name": "Dress",
             "colour_group_name": "Cream", "garment_group_name": "Dresses Ladies", "price": 34.99,
             "explanation": "A cream off-shoulder mini dress adds a statement occasion piece to your otherwise everyday-feminine wardrobe."},
            {"article_id": "0562245005", "prod_name": "Satin Slip Skirt", "product_type_name": "Skirt",
             "colour_group_name": "Dusty Rose", "garment_group_name": "Skirts", "price": 24.99,
             "explanation": "A dusty-rose satin slip skirt brings quiet luxury and occasion versatility without leaving your romantic colour zone."},
        ],
    },
    {
        "customer_id": "DEMO_MINIMAL_004",
        "archetype": "Minimalist",
        "taste_profile": (
            "Your aesthetic is rigorously minimal — a near-monochromatic wardrobe of white, black, and soft grey "
            "across clean-cut basics: crew-neck tees, straight-leg trousers, and fitted knitwear. "
            "You buy deliberately and sparingly, favouring quality staples over seasonal trends. "
            "No prints, no embellishments — pure, quiet confidence."
        ),
        "items": [
            {"article_id": "0791587005", "prod_name": "Slim Crew-neck T-shirt", "product_type_name": "T-shirt",
             "colour_group_name": "White", "garment_group_name": "Jersey Basic", "price": 7.99,
             "explanation": "A clean white crew-neck tee is the cornerstone of your monochromatic, no-fuss wardrobe."},
            {"article_id": "0860345005", "prod_name": "Straight-leg Black Jeans", "product_type_name": "Trousers",
             "colour_group_name": "Black", "garment_group_name": "Trousers", "price": 34.99,
             "explanation": "Straight-leg black jeans are the perfect minimalist bottom — endlessly versatile, timeless, and exactly your palette."},
            {"article_id": "0714790007", "prod_name": "Fine-knit Cashmere-blend Jumper", "product_type_name": "Jumper/Knitwear",
             "colour_group_name": "Greige", "garment_group_name": "Knitwear", "price": 49.99,
             "explanation": "A fine-knit greige jumper adds quiet warmth and luxury to your stripped-back, neutral-only wardrobe."},
            {"article_id": "0706016006", "prod_name": "Relaxed Shirt in Cotton", "product_type_name": "Shirt",
             "colour_group_name": "White", "garment_group_name": "Shirts", "price": 19.99,
             "explanation": "A relaxed white cotton shirt works as a layering piece or standalone — straightforward and precise, like your wardrobe."},
            {"article_id": "0820230007", "prod_name": "Slim Tailored Trousers", "product_type_name": "Trousers",
             "colour_group_name": "Light Grey", "garment_group_name": "Suits & Tailoring", "price": 39.99,
             "explanation": "Light grey tailored trousers bring structure to your minimal look without introducing any unnecessary colour."},
            {"article_id": "0562245006", "prod_name": "Long-line Trench Coat", "product_type_name": "Coat",
             "colour_group_name": "Beige", "garment_group_name": "Coats", "price": 79.99,
             "explanation": "A classic beige trench is the one coat a minimalist wardrobe truly needs — elegant and quietly powerful."},
            {"article_id": "0610776005", "prod_name": "Fitted Black Turtleneck", "product_type_name": "Jumper/Knitwear",
             "colour_group_name": "Black", "garment_group_name": "Knitwear", "price": 24.99,
             "explanation": "A fitted black turtleneck is the most distilled expression of your rigorous, monochrome aesthetic."},
            {"article_id": "0791587006", "prod_name": "Straight-leg White Trousers", "product_type_name": "Trousers",
             "colour_group_name": "White", "garment_group_name": "Trousers", "price": 29.99,
             "explanation": "White straight-leg trousers are a bold-yet-minimal statement that keeps you firmly within your no-colour palette."},
            {"article_id": "0714790008", "prod_name": "Ribbed Tank Top", "product_type_name": "Top",
             "colour_group_name": "Off White", "garment_group_name": "Jersey Basic", "price": 9.99,
             "explanation": "A ribbed off-white tank is a clean, understated layering essential your wardrobe will use constantly."},
            {"article_id": "0860345006", "prod_name": "Cotton-blend Box-fit Jumper", "product_type_name": "Jumper/Knitwear",
             "colour_group_name": "Light Grey", "garment_group_name": "Knitwear", "price": 27.99,
             "explanation": "A light-grey box-fit jumper adds a gentle oversized note while staying true to your soft, neutral palette."},
        ],
    },
    {
        "customer_id": "DEMO_BOLD_005",
        "archetype": "Bold / Colourful",
        "taste_profile": (
            "Your wardrobe is a confident celebration of colour and print — you collect bold patterns, vibrant hues, "
            "and statement pieces with the same enthusiasm others reserve for classics. "
            "From printed co-ords to bright knitwear and patterned dresses, you dress with intent and personality. "
            "Trend engagement is high and you refresh your wardrobe seasonally."
        ),
        "items": [
            {"article_id": "0610776006", "prod_name": "Abstract Print Blouse", "product_type_name": "Blouse",
             "colour_group_name": "Orange/Multicolour", "garment_group_name": "Blouses", "price": 24.99,
             "explanation": "An abstract multicolour blouse is exactly the kind of bold, expressive statement your wardrobe is known for."},
            {"article_id": "0714790009", "prod_name": "Colour-block Knit Jumper", "product_type_name": "Jumper/Knitwear",
             "colour_group_name": "Yellow/Blue", "garment_group_name": "Knitwear", "price": 34.99,
             "explanation": "A yellow and blue colour-block jumper brings the graphic, high-energy aesthetic you consistently reach for."},
            {"article_id": "0791587007", "prod_name": "Wide-leg Printed Trousers", "product_type_name": "Trousers",
             "colour_group_name": "Bright Blue/Pattern", "garment_group_name": "Trousers", "price": 29.99,
             "explanation": "Wide-leg printed trousers match your love of bold pattern with a relaxed silhouette that lets the print lead."},
            {"article_id": "0562245007", "prod_name": "Floral Wrap Midi Dress", "product_type_name": "Dress",
             "colour_group_name": "Pink/Multicolour", "garment_group_name": "Dresses Ladies", "price": 39.99,
             "explanation": "A multicolour floral midi dress adds a dressed-up occasion option that keeps your bold signature intact."},
            {"article_id": "0820230008", "prod_name": "Stripe Co-ord Shorts", "product_type_name": "Shorts",
             "colour_group_name": "Multicolour", "garment_group_name": "Shorts", "price": 19.99,
             "explanation": "Multicolour stripe shorts complete a co-ord outfit while staying firmly in your vibrant, maximalist register."},
            {"article_id": "0706016007", "prod_name": "Tie-dye Cotton T-shirt", "product_type_name": "T-shirt",
             "colour_group_name": "Turquoise/Multicolour", "garment_group_name": "Jersey Fancy", "price": 12.99,
             "explanation": "A tie-dye tee in turquoise and mixed tones is a low-effort way to keep your everyday looks punchy and expressive."},
            {"article_id": "0860345007", "prod_name": "Stripe Knit Cardigan", "product_type_name": "Cardigan",
             "colour_group_name": "Red/Multi", "garment_group_name": "Knitwear", "price": 27.99,
             "explanation": "A red multi-stripe cardigan introduces vintage-inspired colour drama as a layering statement piece."},
            {"article_id": "0714790010", "prod_name": "Patchwork Denim Jacket", "product_type_name": "Jacket",
             "colour_group_name": "Multicolour Denim", "garment_group_name": "Denim", "price": 49.99,
             "explanation": "A patchwork denim jacket is a wearable canvas of colour — the kind of eye-catching piece you gravitate towards."},
            {"article_id": "0791587008", "prod_name": "Bright Yellow Midi Skirt", "product_type_name": "Skirt",
             "colour_group_name": "Bright Yellow", "garment_group_name": "Skirts", "price": 22.99,
             "explanation": "A saturated yellow midi skirt is a bold single-colour statement that anchors any maximalist outfit."},
            {"article_id": "0562245008", "prod_name": "Geometric Jacquard Top", "product_type_name": "Top",
             "colour_group_name": "Blue/Green/Pattern", "garment_group_name": "Jersey Fancy", "price": 19.99,
             "explanation": "A geometric jacquard top adds textural pattern to your existing bold print rotation for a rich, layered look."},
        ],
    },
]


def generate_synthetic_cache():
    """Write synthetic taste profiles, recommendations and explanations to demo_cache/."""
    taste_profiles = {}
    recommendations = {}
    explanations = {}
    curated_ids = []

    for persona in SYNTHETIC_CUSTOMERS:
        cid = persona["customer_id"]
        curated_ids.append(cid)
        taste_profiles[cid] = persona["taste_profile"]
        recommendations[cid] = [
            {k: v for k, v in item.items() if k != "explanation"}
            for item in persona["items"]
        ]
        explanations[cid] = {
            item["article_id"]: item["explanation"]
            for item in persona["items"]
        }

    with open(DEMO_CACHE_DIR / "taste_profiles.json", "w") as f:
        json.dump(taste_profiles, f, indent=2)
    with open(DEMO_CACHE_DIR / "recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)
    with open(DEMO_CACHE_DIR / "explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)
    with open(DEMO_CACHE_DIR / "curated_ids.json", "w") as f:
        json.dump(curated_ids, f, indent=2)

    print(f"✅  Synthetic demo cache written for {len(curated_ids)} personas:")
    for cid in curated_ids:
        print(f"   • {cid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic demo data (no dataset required)")
    parser.add_argument("--live", action="store_true",
                        help="Generate cache from real data + live LLM (requires dataset + ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    import argparse as _ap
    parser2 = _ap.ArgumentParser()
    parser2.add_argument("--synthetic", action="store_true")
    parser2.add_argument("--live", action="store_true")
    args2 = parser2.parse_args()

    if args2.synthetic or not args2.live:
        generate_synthetic_cache()
    else:
        from src.inference import generate_demo_cache, _load_state
        state = _load_state()
        top_customers = (
            state["transactions"]
            .groupby("customer_id")["article_id"]
            .count()
            .sort_values(ascending=False)
            .head(25)
            .index.tolist()
        )
        generate_demo_cache(top_customers)
