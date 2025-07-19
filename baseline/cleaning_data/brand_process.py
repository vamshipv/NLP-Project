import json
import os
import json
import os
import re

"""
Extracts key model variants from reviews and saves them to a JSON file.
This script processes a JSON file containing smartphone reviews, extracts model names,
and generates a list of unique model variants. It handles different formats, including those with
parentheses, and normalizes the model names by removing extra spaces and converting to lowercase.
"""
def normalize_model_variants(model):
    """Returns full model, stripped parentheses, 'Galaxy M51', 'M51', etc."""
    full = model.strip().lower()
    no_parens = re.sub(r"\(.*?\)", "", full).strip()

    variants = set()
    variants.add(full)
    variants.add(no_parens)

    tokens = no_parens.split()
    for i in range(len(tokens)):
        tail = " ".join(tokens[i:])
        if len(tail.split()) >= 1:
            variants.add(tail.strip())

    return list(variants)

"""
Extracts key model variants from smartphone reviews and saves them to a JSON file.
Args:
    reviews_path (str): Path to the input JSON file containing smartphone reviews.
    output_path (str): Path to the output JSON file where model variants will be saved.
"""
def extract_key_model_variants(reviews_path, output_path):
    with open(reviews_path, 'r', encoding='utf-8') as f:
        reviews = json.load(f)

    model_variants = set()

    for review in reviews:
        model_raw = review.get("Model") or review.get("model", "")
        if not model_raw:
            continue

        model_clean = re.sub(r"\s+", " ", model_raw.strip().lower())
        model_with_parens = model_clean

        # Remove anything in parentheses for trimmed version
        model_no_parens = re.sub(r"\(.*?\)", "", model_clean).strip()

        # Get tail (e.g., Galaxy M51) by dropping brand prefix (if present)
        brand = review.get("Brand") or review.get("brand", "")
        brand = brand.strip().lower()

        tail = model_no_parens
        if brand and model_no_parens.startswith(brand):
            tail = model_no_parens[len(brand):].strip()

        # Add desired variants
        model_variants.add(tail)
        model_variants.add(model_no_parens)
        model_variants.add(model_with_parens)

    sorted_variants = sorted(model_variants)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_variants, f, indent=4)

    print(f" Extracted {len(sorted_variants)} model variants to {output_path}")


extract_key_model_variants('../data/reviews.json', '../data/brands.json')