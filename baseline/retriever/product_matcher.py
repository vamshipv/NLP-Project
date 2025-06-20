from rapidfuzz import fuzz, process
import re

class ProductMatcher:
    def __init__(self, chunked_reviews):
        self.chunked_reviews = chunked_reviews
        self.titles = self.extract_titles()

    def extract_titles(self):
        seen = set()
        product_titles = []
        for entry in self.chunked_reviews:
            brand = entry.get("brand", "").strip().lower()
            model = entry.get("model", "").strip().lower()

            # Remove brackets and extra spec info from the model
            model_cleaned = re.sub(r"\(.*?\)", "", model).strip()

            # Avoid duplicate
            if model_cleaned.startswith(brand):
                title = model_cleaned
            else:
                title = f"{brand} {model_cleaned}".strip()

            if title and title not in seen:
                seen.add(title)
                product_titles.append(title)
        return product_titles

    def match(self, query, threshold=90):
        result = process.extractOne(query.lower(), self.titles, scorer=fuzz.token_sort_ratio)
        if result:
            matched_title, score, _ = result
            return matched_title if score >= threshold else None
        return None

    def filter_chunks_by_title(self, matched_core_title):
        pattern = re.compile(re.escape(matched_core_title.lower()))
        return [
            c for c in self.chunked_reviews
            if pattern.search(f"{c.get('brand', '')} {c.get('model', '')}".lower())
        ]
    
    @staticmethod
    def clean_query_for_product_match(query):
    # Remove phrases like "summary on", "reviews about", etc.
        cleaned = re.sub(r"(summary|reviews|feedback)\s+(on|about|for)\s+", "", query.lower()).strip()
        return cleaned
