import pandas as pd
import json
import ast
import re
import html
from rapidfuzz import process, fuzz

class cleaning_merging_data:
    def __init__(self, reviews_path, metadata_path):
        """Initialize file paths and DataFrames"""
        self.reviews_path = reviews_path
        self.metadata_path = metadata_path
        self.reviews_df = None
        self.metadata_df = None
        self.cleaned_df = None

    def load_json_lines(self, file_path):
        """Load JSON lines safely, handling JSON errors"""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    try:
                        parsed_line = ast.literal_eval(line)
                        if isinstance(parsed_line, dict):
                            data.append(parsed_line)
                    except (ValueError, SyntaxError):
                        print(f"Skipping invalid line in {file_path}: {line.strip()}")
        return pd.DataFrame(data)

    def load_data(self):
        """Load reviews and metadata safely, then clean text and remove duplicates"""
        self.metadata_df = self.load_json_lines(self.metadata_path)
        self.metadata_df = self.metadata_df.drop_duplicates(subset=['asin'])
        self.metadata_df = self.metadata_df.dropna(subset=['title'])

        self.reviews_df = self.load_json_lines(self.reviews_path)

        # Ensure required columns exist
        for col in ['asin', 'description', 'title', 'brand']:
            if col not in self.metadata_df.columns:
                self.metadata_df[col] = None

        if 'asin' not in self.reviews_df.columns:
            self.reviews_df['asin'] = None

        # ---- Start Improved Deduplication ----
        self.reviews_df['reviewText'] = self.reviews_df['reviewText'].apply(lambda x: self.clean_text(str(x)))

        # Remove exact duplicates
        self.reviews_df = self.reviews_df.drop_duplicates(subset=['reviewText'], keep='first')

        # Apply fuzzy matching for near-duplicate detection using hashing
        seen_reviews = set()
        unique_reviews = []

        for review in self.reviews_df['reviewText']:
            review_hash = hash(review)  # Hashing for fast comparisons
            if review_hash not in seen_reviews:
                unique_reviews.append(review)
                seen_reviews.add(review_hash)

        self.reviews_df = self.reviews_df[self.reviews_df['reviewText'].isin(unique_reviews)]
        # ---- End Improved Deduplication ----

    def clean_text(self, text):
            """Aggressively clean unwanted characters, HTML artifacts, Unicode sequences, and excessive spaces."""
            if not isinstance(text, str):
                print(f"Warning: Input '{text}' is not a string. Returning empty.")
                return ""

            text = html.unescape(text)

            # Remove common unwanted escape sequences and artifacts
            text = re.sub(r'\\n', ' ', text)  # Replace literal '\n' with spaces
            text = re.sub(r'\\t', ' ', text)  # Replace literal '\t' with spaces
            text = re.sub(r'\\/', '/', text)  # Fix escaped slashes (literal '\/') to '/'
            text = re.sub(r'\\u[\dA-Fa-f]{4}', '', text)  # Remove Unicode escape sequences

            # Remove any remaining Unicode symbols like ®, ©, ™ more comprehensively
            text = re.sub(r'[^\x20-\x7E]+', '', text) # Keep standard printable ASCII, remove others

            # Handle "Description" and similar headers more robustly
            text = re.sub(r'(?:^|\W)(?:description|package includes)(?:$|\W)?', ' ', text, flags=re.IGNORECASE)

            # Consolidate multiple slashes and clean up spaces around them
            text = re.sub(r'\s*/\s*/\s*', ' / ', text)
            text = re.sub(r'/\s*/', '/', text)
            text = re.sub(r'\s*-\s*', ' - ', text)

            # Remove actual newline and tab characters (the control characters)
            text = text.replace("\n", " ").replace("\t", " ")

            # Remove excessive spaces and trim leading/trailing whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text


    @staticmethod
    def simplify_title(title):
        """Shorten product title by removing redundant words and keeping essential details"""
        title = re.sub(r'\b(case|cover|shell|skin)\b', '', title, flags=re.IGNORECASE)  
        title = re.sub(r'\s+', ' ', title).strip()  # Remove excess spaces
        return title

    def clean_metadata(self):
        """Clean metadata, refine title, and include brand only when needed"""
        self.metadata_df['title'] = self.metadata_df['title'].apply(lambda x: self.clean_text(str(x)))
        self.metadata_df['title'] = self.metadata_df['title'].apply(lambda x: self.simplify_title(x))

        # Extract and clean brand if present
        if 'brand' in self.metadata_df.columns:
            self.metadata_df['brand'] = self.metadata_df['brand'].apply(lambda x: self.clean_text(str(x)))
        else:
            self.metadata_df['brand'] = None

        # Decide when to include brand
        def refine_title(row):
            """Only include brand if the title lacks essential details"""
            if row['brand'] and len(row['title'].split()) < 4:  # Example threshold
                return f"{row['brand']} - {row['title']}"
            return row['title']
        
        self.metadata_df['final_title'] = self.metadata_df.apply(refine_title, axis=1)

    def merge_and_clean(self):
        """Merge cleaned reviews with metadata"""
        available_meta_cols = ['asin', 'final_title', 'description', 'brand']
        required_review_cols = ['asin', 'reviewText', 'summary', 'overall']

        if 'asin' in self.reviews_df.columns and 'asin' in self.metadata_df.columns:
            self.cleaned_df = pd.merge(self.metadata_df[available_meta_cols],
                                       self.reviews_df[required_review_cols],
                                       on='asin',
                                       how='inner')
            self.cleaned_df = self.cleaned_df.head(2000)  # Limit for performance

    def save_cleaned_data(self, output_path):
        """Save cleaned dataset to JSON"""
        self.cleaned_df.to_json(output_path, orient="records", indent=4)

# Exampl e Usage
print("Loading the Data")
processor = cleaning_merging_data("reviews_Cell_Phones_and_Accessories.json", "meta_Cell_Phones_and_Accessories.json")
processor.load_data()
print("Loaded the Data")
processor.clean_metadata()
processor.merge_and_clean()
print("Merged and cleaned the Data")
processor.save_cleaned_data("cleaned_amazon_reviews.json")
print("Saved merged and cleaned Data")


