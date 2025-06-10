import os
import pandas as pd
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.combined_texts = []

    def load_csv_and_prepare_texts(self, csv_path):
        # Read the CSV
        df = pd.read_csv(csv_path)

        # Convert all column names to lowercase and strip spaces
        df.columns = [col.strip().lower() for col in df.columns]

        print("Available columns after normalization:", df.columns.tolist())

        # Check if required columns are present
        if 'model' not in df.columns or 'comment' not in df.columns:
            raise KeyError("Required columns 'model' and 'comment' not found in CSV.")

        # Drop rows with missing values and combine fields
        df = df[['model', 'comment']].dropna()
        self.combined_texts = df.apply(
            lambda row: f"Model: {row['model']}. Comment: {row['comment']}", axis=1
        ).tolist()

        return self.combined_texts

    def embed_texts(self):
        """
        Embeds the combined model+comment texts using SentenceTransformer.
        """
        embeddings = self.model.encode(self.combined_texts, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings[0].shape[0])
        self.index.add(np.array(embeddings, dtype=np.float32))

    def save(self, index_path, text_path):
        """
        Saves FAISS index and the combined texts to disk.
        """
        faiss.write_index(self.index, index_path)
        with open(text_path, 'w', encoding='utf-8') as f:
            json.dump(self.combined_texts, f)
    
    def load(self, index_path, text_path):
        """
        Load FAISS index and combined texts from disk.
        """
        self.index = faiss.read_index(index_path)
        with open(text_path, 'r', encoding='utf-8') as f:
            self.combined_texts = json.load(f)

    
    def query(self, question, top_k=3):
        """
        Embed the query and retrieve top_k similar texts.
        
        Args:
            question (str): The user question to query.
            top_k (int): Number of top similar chunks to return.
        
        Returns:
            List of tuples: (score, text)
        """
        if self.index is None or not self.combined_texts:
            raise ValueError("Index or texts are not loaded. Please run embed_texts() first.")

        # Embed the query
        query_embedding = self.model.encode([question], convert_to_numpy=True)

        # Search in the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((dist, self.combined_texts[idx]))

        return results

    
    def print_embeddings(self, num_samples=5):
        """
        Prints embeddings for a few sample texts.
        
        Args:
            num_samples (int): Number of samples to display.
        """
        if not hasattr(self, 'combined_texts'):
            print("No combined texts found. Load and process CSV first.")
            return

        sample_texts = self.combined_texts[:num_samples]
        embeddings = self.model.encode(sample_texts)

        for i, (text, emb) in enumerate(zip(sample_texts, embeddings), 1):
            print(f"\nText {i}: {text}")
            print(f"Embedding shape: {emb.shape}")
            print(f"First 5 values: {emb[:5]}")

    def get_top_chunks(self, question, top_k=3):
        """
        Returns top_k most similar text chunks for a given question.
        """
        query_embedding = self.model.encode([question], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        top_chunks = [self.combined_texts[i] for i in indices[0]]
        return top_chunks



def main():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "../data/final_dataset.csv")
    index_path = os.path.join(current_dir, "reviews_faiss.index")
    text_path = os.path.join(current_dir, "reviews_combined_texts.json")

    retriever = Retriever()
    retriever.load_csv_and_prepare_texts(csv_path)
    retriever.embed_texts()
    retriever.save(index_path, text_path)
    # retriever.print_embeddings()
    print("✅ Embeddings created and saved successfully!")

def interactive_query():
    current_dir = os.path.dirname(__file__)
    index_path = os.path.join(current_dir, "reviews_faiss.index")
    text_path = os.path.join(current_dir, "reviews_combined_texts.json")

    retriever = Retriever()
    retriever.load(index_path, text_path)

    while True:
        question = input("Ask your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        top_results = retriever.query(question, top_k=3)

        print("\nTop results:")
        for score, text in top_results:
            print(f"Score: {score:.4f} — Text: {text}")
        print("\n")

if __name__ == "__main__":
    main()
      # 2. Query the index (run many times)
    # interactive_query()
