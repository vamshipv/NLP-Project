import csv
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

from retriever_gen_sum import Generatorr

class Retriever:
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)
        self.chunks = []
        self.embeddings = None
        self.index = None

    def read_and_chunk_csv(self, file_path, model_col='Model', review_col='comment'):
        """
        Reads a CSV and creates combined text chunks from model + review columns.
        Skips rows where either column is missing or empty.
        """
        max_rows = 10
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if model_col not in reader.fieldnames or review_col not in reader.fieldnames:
                print(f"❌ Columns '{model_col}' or '{review_col}' not found in CSV.")
                return

            count = 0
            for row in reader:
                if count >= max_rows:
                    break
                model = row.get(model_col, '').strip()
                review = row.get(review_col, '').strip()
                if model and review:
                    words = review.split()
                    if len(words) > 50:
                        review = ' '.join(words[:50])
                    self.chunks.append(f"{model}: {review}")
                    count += 1
        # print('✅ Chunks created:', self.chunks)

    def build_index(self):
        """
        Embeds the chunks and builds a FAISS index.
        """
        if not self.chunks:
            print("⚠️ No chunks to embed.")
            return

        self.embeddings = self.model.encode(self.chunks)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings, dtype=np.float32))
        print(f"✅ FAISS index built with {len(self.chunks)} chunks.")

    def search(self, query, top_k=3):
        """
        Searches the index for the top-k most relevant review chunks.
        """
        if self.index is None:
            print("❌ FAISS index not built.")
            return []

        query_emb = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_emb, dtype=np.float32), top_k)
        return [self.chunks[i] for i in indices[0]]



def main():
    retriever = Retriever()
    gen = Generatorr()

    retriever.read_and_chunk_csv("/Users/dechammacg/Documents/NLPPro/specialisation/NLP-Project/baseline/data/final_dataset.csv")
    retriever.build_index()

    user_query = "Summarize the reviews for Samsung Galaxy M51"
    user_query = "what do you think about Samsung Galaxy M51"
    results = retriever.search(user_query)
    # print('results are: ',results)

    all_sentences = []

    for chunk in results:
        # Remove model name prefix if present
        if ":" in chunk:
            _, review = chunk.split(":", 1)
        else:
            review = chunk

        # Tokenize into sentences
        sentences = sent_tokenize(review)

        for sent in sentences:
            clean_sent = sent.strip()
            if len(clean_sent.split()) > 4 and not clean_sent.lower().startswith("here are some"):
                all_sentences.append(clean_sent)
    print('all_sentences',all_sentences)

    summary = gen.summarize_reviews(all_sentences)
    print("Summary:", summary)
    
    reviews = [
        "Camera quality is excellent and battery backup is really good.",
        "The phone heats up during charging and the UI is confusing.",
        "Build quality is cheap and it's not water-resistant.",
        "Dolby sound only works with headphones, not internal speakers.",
        "Display is vibrant, zoom is great and the macro camera works well.",
    ]

    # summary = gen.summarize_reviews(reviews)
    # print("Summary:", summary)


if __name__ == "__main__":
    main()
