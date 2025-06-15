import nltk
import json
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy

class ReviewChunkIndexer:
    def __init__(self, reviews_df):
        nltk.download('punkt')
        self.reviews_df = reviews_df
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.nlp = spacy.load("en_core_web_sm")
        self.index = None
        self.chunked_reviews = []

    def compact_title(self, full_title):
        # Compact the title to first part before " for " or max 6 words
        if not isinstance(full_title, str):
            return ""
        compact = full_title.split(" for ")[0]
        return " ".join(compact.split()[:6])

    def chunk_reviews(self):
        """Chunk reviews and include ASIN, brand, compact title"""
        chunked_reviews = []
        seen_chunks = set()

        for _, row in self.reviews_df.iterrows():
            review_text = str(row.get("reviewText", "")).strip()
            brand = str(row.get("brand", "")).strip()
            asin = str(row.get("asin", "")).strip()
            full_title = str(row.get("final_title", "")).strip()
            compact_title = self.compact_title(full_title)

            if not review_text:
                continue

            # Sentence-level chunking
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
                            "asin": asin,
                            "compact_title": compact_title
                        })
                        seen_chunks.add(clean_chunk)
                    chunk = sentence
            # Add last chunk
            clean_chunk = chunk.strip()
            if clean_chunk and clean_chunk not in seen_chunks:
                chunked_reviews.append({
                    "text": clean_chunk,
                    "brand": brand,
                    "asin": asin,
                    "compact_title": compact_title
                })
                seen_chunks.add(clean_chunk)

        # Save
        with open("chunked_reviews.json", "w", encoding="utf-8") as f:
            json.dump(chunked_reviews, f, indent=4)

        self.chunked_reviews = chunked_reviews
        return chunked_reviews

    def index_chunks(self):
        """Create FAISS vector index for chunks"""
        if not self.chunked_reviews:
            raise ValueError("No chunked reviews to index. Run chunk_reviews first.")
        
        embeddings = np.array([self.model.encode(entry["text"]) for entry in self.chunked_reviews])
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        faiss.write_index(self.index, "reviews.index")

        print(f"✅ Indexed {len(self.chunked_reviews)} chunks in FAISS")

    def retrieve_chunks(self, query, top_k=5):
        """Retrieve top_k chunks relevant to query"""
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec), top_k)
        results = [self.chunked_reviews[i] for i in I[0]]
        return results

# Example  Usage
df = pd.read_json("cleaned_amazon_reviews.json")
indexer = ReviewChunkIndexer(df)
indexer.chunk_reviews()
indexer.index_chunks()
print("Chunking saved in 'chunked_reviews.json' and FAISS index stored in 'reviews.index'")