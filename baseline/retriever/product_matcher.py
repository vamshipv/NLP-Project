from rapidfuzz import fuzz, process
import re

"""
ProductMatcher is a class that matches product titles from chunked reviews based on a query.
It extracts unique product titles from the reviews, matches a query against these titles,
and filters the reviews based on the matched title.
It also provides a method to clean the query for better matching.
It is designed to work with chunked reviews, where each review contains a brand and model.
Currently, it uses fuzzy matching to find the best match for a given query and also work in progress
to clean the query by removing unnecessary phrases.
"""

class ProductMatcher:
    def __init__(self, chunked_reviews):
        self.chunked_reviews = chunked_reviews
        self.titles = self.extract_titles()

    """
    This method extracts unique product titles from the chunked reviews.
    It combines the brand and model, cleans the model by removing brackets and extra spec info,
    and ensures that the titles are unique.
    """
    def extract_titles(self):
        seen = set()
        product_titles = []
        for entry in self.chunked_reviews:
            brand = entry.get("brand", "").strip().lower()
            model = entry.get("model", "").strip().lower()

            model_cleaned = re.sub(r"\(.*?\)", "", model).strip()

            if model_cleaned.startswith(brand):
                title = model_cleaned
            else:
                title = f"{brand} {model_cleaned}".strip()

            if title and title not in seen:
                seen.add(title)
                product_titles.append(title)
        return product_titles

    """
    This method matches a query against the extracted product titles using fuzzy matching.
    It returns the matched title if the score is above a specified threshold.
    """
    def match(self, query, threshold=90):
        result = process.extractOne(query.lower(), self.titles, scorer=fuzz.token_sort_ratio)
        if result:
            matched_title, score, _ = result
            return matched_title if score >= threshold else None
        return None

    """
    This method filters the chunked reviews based on the matched core title.
    It searches for the matched title in the brand and model fields of each review.
    """
    def filter_chunks_by_title(self, matched_core_title):
        pattern = re.compile(re.escape(matched_core_title.lower()))
        return [
            c for c in self.chunked_reviews
            if pattern.search(f"{c.get('brand', '')} {c.get('model', '')}".lower())
        ]

    """
    This method cleans the query by removing phrases like "summary on", "reviews about", etc.
    It returns the cleaned query for better matching.
    #TODO
    Currently, working in progress.
    """   
    @staticmethod
    def clean_query_for_product_match(query):
        cleaned = re.sub(r"(summary|reviews|feedback)\s+(on|about|for)\s+", "", query.lower()).strip()
        return cleaned
