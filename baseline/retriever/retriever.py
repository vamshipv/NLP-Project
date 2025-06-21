import nltk
import json
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
import os
import logging
from datetime import datetime
from product_matcher import ProductMatcher
import re

"""
This script is a retriever module that processes product reviews, chunks them into manageable pieces, 
indexes them using FAISS, and retrieves relevant chunks based on user queries.
"""

# File paths
json_file = os.path.join("..", "data", "reviews.json")
output_dir = os.path.join("..", "data")
os.makedirs(output_dir, exist_ok=True)
chunked_path = os.path.join(output_dir, "chunked_reviews.json")
index_path = os.path.join(output_dir, "reviews.index")

# Logging configuration not workng
log_dir = os.path.join("..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "retriever_log.json")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")

""""
This retriever module is designed to:
1. Load product reviews from a JSON file.
2. Chunk the reviews into smaller pieces to fit within a specified token limit.
3. Index these chunks using FAISS for efficient retrieval.
4. Retrieve relevant chunks based on user queries, matching them to product titles.
"""

class Retriever:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        self.reviews_df = pd.read_json(json_file)
        self.model = SentenceTransformer("intfloat/e5-base-v2")
        self.nlp = spacy.load("en_core_web_sm")
        self.index = None
        self.chunked_reviews = []

    """
    This method processes the reviews DataFrame, chunks the text into manageable pieces, and saves them to a JSON file.
    It ensures that each chunk is unique and does not exceed a specified length (512 characters).
    Returns:
        list: A list of dictionaries containing the chunked review text, brand, model, and stars.
    """
    def chunk_reviews(self):
        chunked_reviews = []
        seen_chunks = set()

        for _, row in self.reviews_df.iterrows():
            review_text = str(row.get("comment", "")).strip()
            brand = str(row.get("Brand", "")).strip()
            model = str(row.get("Model", "")).strip()
            stars = str(row.get("stars", "")).strip()

            if not review_text:
                continue

            sentences = nltk.sent_tokenize(review_text)
            chunk = ""
            for sentence in sentences:
                if len(chunk) + len(sentence) < 512:
                    chunk += " " + sentence
                else:
                    clean_chunk = chunk.strip()
                    if clean_chunk and clean_chunk not in seen_chunks:
                        chunked_reviews.append({
                            "text": clean_chunk,
                            "brand": brand,
                            "model": model,
                            "stars": stars
                        })
                        seen_chunks.add(clean_chunk)
                    chunk = sentence
            clean_chunk = chunk.strip()
            if clean_chunk and clean_chunk not in seen_chunks:
                chunked_reviews.append({
                    "text": clean_chunk,
                    "brand": brand,
                    "model": model,
                    "stars": stars
                })
                seen_chunks.add(clean_chunk)

        with open(chunked_path, "w", encoding="utf-8") as f:
            json.dump(chunked_reviews, f, indent=4)

        self.chunked_reviews = chunked_reviews

        logging.info(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "event": "chunk_reviews",
            "num_chunks": len(chunked_reviews),
            "output_file": chunked_path
        }))

        return chunked_reviews

    """
    This method indexes the chunked reviews using FAISS.
    It encodes the text of each chunk into embeddings and adds them to a FAISS index for efficient similarity search.
    """
    def index_chunks(self):
        if not self.chunked_reviews:
            raise ValueError("No chunked reviews to index. Run chunk_reviews first.")

        embeddings = np.array([self.model.encode(entry["text"]) for entry in self.chunked_reviews], show_progress_bar=False)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        faiss.write_index(self.index, index_path)

        logging.info(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "event": "index_chunks",
            "num_chunks": len(self.chunked_reviews),
            "index_file": index_path
        }))

        print(f"Indexed {len(self.chunked_reviews)} chunks in FAISS")

    """
    This method retrieves the embedding for a given text using the SentenceTransformer model.
    """
    def get_embedding(self, text):
        return self.model.encode([text])[0].reshape(1, -1).astype("float32")

    """
    This method retrieves relevant chunks based on a user query.
    It matches the query to product titles, filters the chunks by the matched title, and retrieves the top_k relevant chunks.

    #TODO
    Working in progess:
    - It uses the ProductMatcher to find the best matching product title based on the query.
    - It filters the chunks based on the matched title.
    - It uses FAISS to search for the top_k most relevant chunks based on the query embedding.
    - Currently, exact matching is not implemented, but it retrieves chunks based on the closest match to the query.
    """
    def retrieve(self, query, top_k=15):
        with open(chunked_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        matcher = ProductMatcher(chunks)
        print(matcher.titles)
        query_for_matching = matcher.clean_query_for_product_match(query)
        print(f"Query for matching: {query_for_matching}")
        matched_title = matcher.match(query_for_matching)
        print(f"Matched title: {matched_title}")

        if not matched_title:
            logging.info(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "event": "retrieve",
                "query": query,
                "matched_title": None,
                "returned_chunks": 0
            }))
            return []
        if not matched_title:
            return []
        matched_core = re.sub(r"\(.*?\)", "", matched_title).strip().lower()
        filtered_chunks = matcher.filter_chunks_by_title(matched_core)

        if not filtered_chunks:
            logging.info(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "event": "retrieve",
                "query": query,
                "matched_title": matched_title,
                "returned_chunks": 0
            }))
            return []

        embedding_dim = self.get_embedding("sample").shape[1]
        temp_index = faiss.IndexFlatL2(embedding_dim)
        review_embeddings = np.vstack([self.get_embedding(f"passage: {c['text']}") for c in filtered_chunks])
        temp_index.add(review_embeddings)

        query_vec = self.get_embedding(f"query: {query_for_matching}")
        _, I = temp_index.search(query_vec, min(top_k, len(filtered_chunks)))
        results = [filtered_chunks[i] for i in I[0]]

        logging.info(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "event": "ui_retrieve",
            "query": query,
            "matched_product": matched_title or "None",
            "returned_chunks": len(results)
        }))

        return results


if __name__ == "__main__":
    indexer = Retriever()
    indexer.chunk_reviews()
    indexer.index_chunks()
    print("Chunking and FAISS index are complete.")
