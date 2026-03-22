"""
Fine-tune dataset preparation.
Generates a JSONL file in chat format (compatible with OpenAI / Groq fine-tuning).
Run: python -m assistant.finetune_prep
"""

import json
import pandas as pd

PRODUCTS_CSV = "data/products.csv"
OUTPUT_JSONL = "data/finetune_dataset.jsonl"

SYSTEM_MSG = "You are ShopBot, a helpful AI shopping assistant for an e-commerce store."

# Template Q&A pairs per product
def generate_pairs(row):
    name = row["name"]
    price = row["price"]
    desc = row["description"]
    category = row["category"]
    rating = row["rating"]
    stock = row["stock"]

    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": f"Tell me about {name}"},
                {"role": "assistant", "content": f"{name} is a {category} product priced at ${price}. {desc} It has a rating of {rating}/5 with {stock} units in stock."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": f"Is {name} worth buying?"},
                {"role": "assistant", "content": f"Yes! {name} is highly rated at {rating}/5 stars. {desc} At ${price}, it offers great value in the {category} category."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": f"How much does {name} cost?"},
                {"role": "assistant", "content": f"{name} is priced at ${price}. We currently have {stock} units available."},
            ]
        },
    ]


def build_finetune_dataset():
    df = pd.read_csv(PRODUCTS_CSV)
    all_pairs = []

    for _, row in df.iterrows():
        all_pairs.extend(generate_pairs(row))

    # Add general shopping Q&A
    general_pairs = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": "What categories do you have?"},
                {"role": "assistant", "content": "We carry products across Electronics, Clothing, Shoes, Kitchen, Books, Sports, Beauty, Health, and Accessories. What are you looking for today?"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": "I have a budget of $50, what can you recommend?"},
                {"role": "assistant", "content": "Great! With a $50 budget I'd suggest checking out our Yoga Mat ($35.99), Running Shorts ($34.99), Vitamin C Serum ($24.99), or The Alchemist book ($14.99). What category interests you most?"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": "What's your best rated product?"},
                {"role": "assistant", "content": "Our top-rated products are The Alchemist and Harry Potter Box Set, both rated 4.9/5! In electronics, the Apple AirPods Pro leads with 4.8/5."},
            ]
        },
    ]
    all_pairs.extend(general_pairs)

    with open(OUTPUT_JSONL, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Generated {len(all_pairs)} training examples -> {OUTPUT_JSONL}")


if __name__ == "__main__":
    build_finetune_dataset()
