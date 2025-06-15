import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from generator import Generator

class ReviewRetriever:
    def __init__(self, chunk_file="chunked_reviews.json", faiss_index_file="reviews.index"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(faiss_index_file)
        with open(chunk_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(self, query, top_k=15):
        # Encode query and search in FAISS
        query_embedding = self.model.encode(query).reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return results

def main():
    user_query = input("Enter your product query: ")

    # Retrieve  relevant chunks (up to 15 to avoid too long inputs)
    retriever = ReviewRetriever()
    retrieved_chunks = retriever.retrieve(user_query, top_k=15)

    if not retrieved_chunks:
        print("No relevant reviews found.")
        return

    # Generate abstractive paraphrased summary
    generator = Generator()
    summary = generator.generate_summary(user_query, retrieved_chunks)

    print("\n--- Generated Summary ---")
    print(summary)

if __name__ == "__main__":
    main()
