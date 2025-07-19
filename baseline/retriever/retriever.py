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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


"""
This script is a retriever module that processes product reviews, chunks them into manageable pieces, 
indexes them using FAISS, and retrieves relevant chunks based on user queries.
"""

# File paths
json_file = os.path.join("..", "data", "reviews.json")
output_dir = os.path.join("..", "data")
os.makedirs(output_dir, exist_ok=True)
chunked_path = os.path.join(output_dir, "general_chunks.json")
index_path = os.path.join(output_dir, "general_chunks.index")
aspect_index_dir = os.path.join(output_dir, "aspect_indexes")
os.makedirs(aspect_index_dir, exist_ok=True)

log_dir = os.path.join("..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "summary_log.json")
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
        self.aspect_keywords = {
            "battery": ["battery", "charge", "charging", "mah", "power", "drain"],
            "camera": ["camera", "photo", "picture", "lens", "image", "zoom", "video"],
            "performance": ["lag", "smooth", "fast", "slow", "processor", "snapdragon", "performance"],
            "display": ["screen", "display", "brightness", "resolution", "refresh rate", "touch"],
            "build": ["build", "design", "material", "durability", "weight", "feel"],
            "software": ["ui", "os", "update", "bloatware", "interface", "android", "software"],
            "heating": ["heat", "heating", "warm", "temperature", "overheat"]
        }

    """
    This method retrieves the embedding for a given text using the SentenceTransformer model.
    It encodes the text into a vector representation suitable for similarity search.
    """

    def get_embedding(self, text):
        return self.model.encode([text])[0].reshape(1, -1).astype("float32")
    
    """
    This method processes the reviews DataFrame, chunks the text into manageable pieces, and saves them to a JSON file.
    It ensures that each chunk is unique and does not exceed a specified length (512 characters).
    Returns:
        list: A list of dictionaries containing the chunked review text, brand, model, and stars.
    """
    def chunk_reviews(self):
        chunked_reviews = []
        seen_chunks = set()
        nlp = spacy.load("en_core_web_sm")

        for _, row in self.reviews_df.iterrows():
            review_text = str(row.get("comment", "")).strip()
            brand = str(row.get("Brand", "")).strip()
            model = str(row.get("Model", "")).strip()
            stars = str(row.get("stars", "")).strip()

            if not review_text:
                continue

            # Sentence segmentation
            doc = nlp(review_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not sentences:
                continue

            

            # General chunking (512 char limit)
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

            if len(sentences) < 2:
                # Treat the single sentence as a standalone chunk
                chunk_text = sentences[0]
                for aspect, keywords in self.aspect_keywords.items():
                    if any(k in chunk_text.lower() for k in keywords):
                        chunked_reviews.append({
                            "text": chunk_text,
                            "brand": brand,
                            "model": model,
                            "stars": stars,
                            "aspect": aspect
                        })
                        seen_chunks.add(chunk_text)
                continue

            # Aspect-based clustering
            sentence_embeddings = self.model.encode(sentences, normalize_embeddings=True)
            num_clusters = min(len(sentences), 5)
            clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='average')
            labels = clustering.fit_predict(sentence_embeddings)

            clusters = {}
            for idx, label in enumerate(labels):
                clusters.setdefault(label, []).append((sentences[idx], sentence_embeddings[idx]))

            for aspect, keywords in self.aspect_keywords.items():
                aspect_desc = f"passage: {aspect} related features in mobile devices"
                aspect_embedding = self.model.encode([aspect_desc], normalize_embeddings=True)[0]

                for cluster_sentences in clusters.values():
                    cluster_texts = [s[0] for s in cluster_sentences]
                    cluster_embedding = np.mean([s[1] for s in cluster_sentences], axis=0).reshape(1, -1)
                    similarity = cosine_similarity(cluster_embedding, aspect_embedding.reshape(1, -1))[0][0]

                    # Filter sentences within the cluster
                    relevant_sentences = [
                        s for s in cluster_texts
                        if any(k in s.lower() for k in keywords)
                    ]

                    # Use filtered sentences if they exist and similarity is high
                    if similarity >= 0.5 and relevant_sentences:
                        chunk_text = " ".join(relevant_sentences).strip()
                    # Otherwise, fallback to full cluster if similarity is strong or keyword match is weak but present
                    elif similarity >= 0.5 or any(any(k in s.lower() for k in keywords) for s in cluster_texts):
                        chunk_text = " ".join(cluster_texts).strip()
                    else:
                        continue  # Skip this cluster if it's not relevant

                    if chunk_text and chunk_text not in seen_chunks:
                        chunked_reviews.append({
                            "text": chunk_text,
                            "brand": brand,
                            "model": model,
                            "stars": stars,
                            "aspect": aspect
                        })
                        seen_chunks.add(chunk_text)

        # Save to file
        with open(chunked_path, "w", encoding="utf-8") as f:
            json.dump(chunked_reviews, f, indent=4)

        self.chunked_reviews = chunked_reviews
        return chunked_reviews


    """    
    This method indexes the chunked reviews using FAISS.
    It creates a general index for non-aspect chunks and separate indexes for each aspect.
    It saves the indexes to disk and logs the indexing process.
    """
    def index_chunks(self):
        if not self.chunked_reviews:
            raise ValueError("No chunked reviews to index. Run chunk_reviews first.")

        # Index general (non-aspect) chunks
        general_chunks = [c for c in self.chunked_reviews if "aspect" not in c]
        general_texts = [f"passage: {c['text']}" for c in general_chunks]
        embeddings = self.model.encode(general_texts, convert_to_numpy=True, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        faiss.write_index(self.index, index_path)

        with open(os.path.join(output_dir, "general_chunks.json"), "w", encoding="utf-8") as f:
            json.dump(general_chunks, f, indent=2)

        # Index each aspect separately
        for aspect in self.aspect_keywords:
            aspect_chunks = [c for c in self.chunked_reviews if c.get("aspect") == aspect]
            if not aspect_chunks:
                continue

            aspect_dir = os.path.join(aspect_index_dir, aspect)
            os.makedirs(aspect_dir, exist_ok=True)

            with open(os.path.join(aspect_dir, f"{aspect}_chunks.json"), "w", encoding="utf-8") as f:
                json.dump(aspect_chunks, f, indent=2)

            texts = [f"passage: {c['text']}" for c in aspect_chunks]
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            faiss.write_index(index, os.path.join(aspect_dir, f"{aspect}.index"))



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
    def retrieve(self, query, top_k=20):
        chunk_path = os.path.join(output_dir, "general_chunks.json")

        if not os.path.exists(chunk_path):
            print("General chunk file not found.")
            return []

        with open(chunk_path, "r", encoding="utf-8") as f:
            general_chunks = json.load(f)

        matcher = ProductMatcher()
        matched_title = matcher.match_brand(query)
        print(f"Matched title: {matched_title}")
        if not matched_title:
            return []

        # Filter by brand/model BEFORE indexing
        pattern = re.compile(rf"\b{re.escape(matched_title.lower())}\b")
        filtered_chunks = [
            c for c in general_chunks
            if pattern.search(f"{c.get('brand', '')} {c.get('model', '')}".lower())
        ]

        if not filtered_chunks:
            return []

        # Build temporary FAISS index from filtered chunks
        texts = [f"passage: {c['text']}" for c in filtered_chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        temp_index = faiss.IndexFlatL2(dim)
        temp_index.add(embeddings)

        query_vec = self.get_embedding(f"query: {query}")
        _, I = temp_index.search(query_vec, min(top_k, len(filtered_chunks)))
        results = [filtered_chunks[i] for i in I[0]]
        results = self.remove_duplicate_chunks(results)
        # print(f"Retrieved after duplicate removal: {results}")
        return results

    """
    This method retrieves relevant chunks based on a user query and aspect.
    It matches the query to product titles, filters the chunks by the matched title, and retrieves the top_k relevant chunks.

    #TODO
    Working in progess:
    - It uses the ProductMatcher to find the best matching product title based on the query.
    - It filters the chunks based on the matched title.
    - It uses FAISS to search for the top_k most relevant chunks based on the query embedding.
    - Currently, exact matching is not implemented, but it retrieves chunks based on the closest match to the query.
    """
    def retrieve_by_aspect(self, query, aspect, top_k=40):
        chunk_path = os.path.join(aspect_index_dir, aspect, f"{aspect}_chunks.json")

        if not os.path.exists(chunk_path):
            print(f"Chunk file for aspect '{aspect}' not found.")
            return []

        with open(chunk_path, "r", encoding="utf-8") as f:
            aspect_chunks = json.load(f)

        matcher = ProductMatcher()
        matched_title = matcher.match_brand(query)
        # print(f"Matched title: {matched_title}")
        if not matched_title:
            return []

        # Filter by brand/model BEFORE indexing
        pattern = re.compile(rf"\b{re.escape(matched_title.lower())}\b")
        filtered_chunks = [
            c for c in aspect_chunks
            if pattern.search(f"{c.get('brand', '')} {c.get('model', '')}".lower())
        ]

        if not filtered_chunks:
            return []

        # Build temporary FAISS index from filtered aspect chunks
        texts = [f"passage: {c['text']}" for c in filtered_chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        temp_index = faiss.IndexFlatL2(dim)
        temp_index.add(embeddings)

        query_vec = self.get_embedding(f"query: {query}")
        _, I = temp_index.search(query_vec, min(top_k, len(filtered_chunks)))
        results = [filtered_chunks[i] for i in I[0]]
        results = self.remove_duplicate_chunks(results)
        # print(f"retrieved after duplicate {results}")
        # results = self.filter_sentences_by_aspect(results, aspect)
        # print(f"Filtered sentences by aspect '{aspect}': {results}")
        return results
    
    """
    This method removes duplicate chunks from a list of chunks.
    It checks for unique text content in the chunks and returns a list of unique chunks.
    """
    def remove_duplicate_chunks(self, chunks):
        seen = set()
        unique = []
        for c in chunks:
            text = c["text"].strip()
            if text not in seen:
                seen.add(text)
                unique.append(c)
        return unique
    
    """
    This method filters sentences in the chunks based on the aspect keyword.
    Moved to User_query_process module - Work in progress
    """
    def filter_sentences_by_aspect(self, chunks, aspect_keyword):
        aspect_sentences = []
        for chunk in chunks:
            sentences = nltk.sent_tokenize(chunk["text"])
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in aspect_keyword):
                    aspect_sentences.append(sentence.strip())
        return aspect_sentences

if __name__ == "__main__":
    retriever = Retriever()
    retriever.chunk_reviews()
    retriever.index_chunks()
    print("Chunking and FAISS indexing complete.")
