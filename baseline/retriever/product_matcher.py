from flashtext import KeywordProcessor
import json
import re
import os

output_dir = os.path.join("..", "data")
os.makedirs(output_dir, exist_ok=True)
brands_path = os.path.join(output_dir, "brands.json")

"""
This module is designed to match product brands from user queries using the FlashText library.
It uses a keyword processor to efficiently search for brand names in user input.
It can also clean the query by removing filler words and punctuation to isolate the brand name.
It is initialized with a path to a JSON file containing brand keywords.
The `ProductMatcher` class provides methods to clean the user query and match it against the brand keywords.
The `clean_query_for_brand_match` method strips filler words and punctuation from the user query,
and the `match_brand` method extracts the brand name from the cleaned query.
"""
class ProductMatcher:
    def __init__(self, brand_file_path=brands_path):
        self.keyword_processor = KeywordProcessor(case_sensitive=False)

        if os.path.exists(brand_file_path):
            with open(brand_file_path, "r", encoding="utf-8") as f:
                brand_keywords = json.load(f)
            self.keyword_processor.add_keywords_from_list(brand_keywords)
        else:
            raise FileNotFoundError(f"Brand file not found: {brand_file_path}")

    """
    Initialize the ProductMatcher with a brand file path.
    The brand file should contain a list of brand names to match against user queries.
    The brand names are loaded from a JSON file and added to the keyword processor.
    """
    def clean_query_for_brand_match(self, query):
        """Strip filler words and punctuation from user query to isolate brand."""
        query = query.lower()
        query = re.sub(r"(summar(y|ize|ise)|reviews|feedback|opinions)\s+(on|about|for)?", "", query)
        query = re.sub(r"(please|kindly|can you|what do you think about|tell me about|i want|should i buy)", "", query)
        query = re.sub(r"[^\w\s\-]", "", query)  # remove punctuation
        query = re.sub(r"\s+", " ", query).strip()
        return query

    """
    This method matches the brand name from the user query using FlashText.
    It cleans the query to remove filler words and punctuation,
    and then extracts the brand name using the keyword processor.
    Extract brand name from query using FlashText.
    If a brand name is found, it returns the first match; otherwise, it returns None.
    """
    def match_brand(self, query):
        cleaned = self.clean_query_for_brand_match(query)
        matches = self.keyword_processor.extract_keywords(cleaned)
        return matches[0] if matches else None
