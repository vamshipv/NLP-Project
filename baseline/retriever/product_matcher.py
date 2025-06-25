from flashtext import KeywordProcessor
import json
import re
import os

output_dir = os.path.join("..", "data")
os.makedirs(output_dir, exist_ok=True)
brands_path = os.path.join(output_dir, "brands.json")

class ProductMatcher:
    def __init__(self, brand_file_path=brands_path):
        self.keyword_processor = KeywordProcessor(case_sensitive=False)

        if os.path.exists(brand_file_path):
            with open(brand_file_path, "r", encoding="utf-8") as f:
                brand_keywords = json.load(f)
            self.keyword_processor.add_keywords_from_list(brand_keywords)
        else:
            raise FileNotFoundError(f"Brand file not found: {brand_file_path}")

    def clean_query_for_brand_match(self, query):
        """Strip filler words and punctuation from user query to isolate brand."""
        query = query.lower()
        query = re.sub(r"(summar(y|ize|ise)|reviews|feedback|opinions)\s+(on|about|for)?", "", query)
        query = re.sub(r"(please|kindly|can you|what do you think about|tell me about|i want|should i buy)", "", query)
        query = re.sub(r"[^\w\s\-]", "", query)  # remove punctuation
        query = re.sub(r"\s+", " ", query).strip()
        return query

    def match_brand(self, query):
        """Extract brand name from query using FlashText."""
        cleaned = self.clean_query_for_brand_match(query)
        matches = self.keyword_processor.extract_keywords(cleaned)
        return matches[0] if matches else None
