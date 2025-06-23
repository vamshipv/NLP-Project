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
from transformers import pipeline

# Define aspect keywords
ASPECT_KEYWORDS = {
    "battery": ["battery", "charge", "power", "backup"],
    "display": ["display", "screen", "brightness", "resolution"],
    "ui": ["ui", "interface", "user interface", "navigation"],
    "performance": ["performance", "lag", "smooth", "responsive", "speed", "multitask"],
    "sound effect": ["sound", "audio", "volume", "speaker"],
    "camera": ["camera", "photo", "image", "video", "lens"],
    "price": ["price", "cost", "expensive", "cheap", "value"],
    "processor": ["processor", "chip", "cpu", "snapdragon", "mediatek"],
    "speed": ["speed", "slow", "fast", "quick"]
}

# File paths
json_file = os.path.join("..", "data", "reviews.json")
output_dir = os.path.join("..", "data")
os.makedirs(output_dir, exist_ok=True)
chunked_path = os.path.join(output_dir, "chunked_reviews.json")
index_path = os.path.join(output_dir, "reviews.index")

# Logging configuration
log_dir = os.path.join("..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "retriever_log.json")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")


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

    def index_chunks(self):
        if not self.chunked_reviews:
            raise ValueError("No chunked reviews to index. Run chunk_reviews first.")

        embeddings = np.array([self.model.encode(entry["text"]) for entry in self.chunked_reviews])
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

        print(f"✅ Indexed {len(self.chunked_reviews)} chunks in FAISS")

    def get_embedding(self, text):
        return self.model.encode([text])[0].reshape(1, -1).astype("float32")

    def filter_by_aspect(self, chunks, aspect_keywords):
        return [chunk for chunk in chunks if any(keyword in chunk["text"].lower() for keyword in aspect_keywords)]

    def retrieve(self, query, top_k=15, summarize=False):
        with open(chunked_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        matcher = ProductMatcher(chunks)
        query_for_matching = matcher.clean_query_for_product_match(query)
        matched_title = matcher.match(query_for_matching)

        if not matched_title:
            print(f"❌ No match found for: {query}")
            return []

        matched_core = re.sub(r"\(.*?\)", "", matched_title).strip().lower()
        filtered_chunks = matcher.filter_chunks_by_title(matched_core)

        if not filtered_chunks:
            print(f"❌ No chunks found for: {matched_title}")
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
            "event": "retrieve",
            "query": query,
            "matched_product": matched_title,
            "returned_chunks": len(results)
        }))

        # Aspect grouping
        aspect_chunks = {}
        for aspect, keywords in ASPECT_KEYWORDS.items():
            aspect_filtered = self.filter_by_aspect(results, keywords)
            if aspect_filtered:
                aspect_chunks[aspect] = aspect_filtered

        for aspect, chunks in aspect_chunks.items():
            print(f"\n📌 Aspect: {aspect.capitalize()} ({len(chunks)} comments)")
            for i, c in enumerate(chunks[:3]):
                print(f" - {i+1}. {c['text'][:100]}...")

        # Optional summarization (if enabled)
        if summarize:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            for aspect, chunks in aspect_chunks.items():
                combined = " ".join([c["text"] for c in chunks])
                if len(combined.split()) > 100:
                    combined = " ".join(combined.split()[:100])
                summary = summarizer(combined, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
                print(f"\n📝 Summary for {aspect}:\n{summary}")

        return results


if __name__ == "__main__":
    retriever = Retriever()
    retriever.chunk_reviews()
    retriever.index_chunks()
    query = input("🔎 Enter product name to retrieve reviews: ")
    retriever.retrieve(query, top_k=15, summarize=True)
