import csv
import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from gen_ollama import summarize_with_ollama
from gen_ollama_deepseek import summarize_with_deepseek
from gen_ollama_gemma import summarize_with_gemma
from gen_ollama_phi import summarize_with_phi
# nltk.download('punkt')

class Retriever:
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)
        self.chunks = []
        self.embeddings = None
        self.index = None

        self.filler_patterns = [
            r"i am giving this review.*?\.",
            r"here are some.*?\.",
            r"just bought.*?\.",
            r"i have been using.*?\.",
            r"after [0-9]+ (days|weeks|months).*?\.",
            r"reviewed after.*?\.",
        ]

    def read_and_chunk_csv(self, file_path, model_col='Model', review_col='comment'):
        max_rows = 800
        seen = set()
        for_chunks = []

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

                if not model or not review:
                    continue

                clean_review = self.clean_review_text(review)
                if len(clean_review.split()) < 10:
                    continue  # skip too short

                # Truncate after cleaning
                words = clean_review.split()
                if len(words) > 50:
                    clean_review = ' '.join(words[:50])

                key = (model.lower(), clean_review.lower())
                if key in seen:
                    continue
                seen.add(key)

                self.chunks.append(f"{model}: {clean_review}")
                count += 1


    def build_index(self):
        if not self.chunks:
            print("⚠️ No chunks to embed.")
            return
        self.embeddings = self.model.encode(self.chunks)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings, dtype=np.float32))
        print(f"✅ FAISS index built with {len(self.chunks)} chunks.")

    def search(self, query, top_k=5, match_names=True):
        if self.index is None:
            print("❌ FAISS index not built.")
            return []

        if match_names:
            product_names = list({chunk.split(":")[0] for chunk in self.chunks if ":" in chunk})
            name_vecs = self.model.encode(product_names)
            query_vec = self.model.encode([query])
            sims = cosine_similarity(query_vec, name_vecs)[0]
            best_idx = np.argmax(sims)

            if sims[best_idx] >= 0.6:
                matched_name = product_names[best_idx]
                print(f"🔍 Matched '{query}' to product name: '{matched_name}'")
                query = matched_name

        query_emb = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_emb, dtype=np.float32), top_k * 2)

        seen_reviews = set()
        results = []
        for i in indices[0]:
            chunk = self.chunks[i]
            review = chunk.split(":", 1)[-1].strip().lower()
            if review not in seen_reviews:
                seen_reviews.add(review)
                results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    def clean_review_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[_\-]+', ' ', text)
        text = re.sub(r'\.\.+', '.', text)
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
        text = re.sub(r'[^a-z0-9.,!? ]+', '', text)

        for pattern in self.filler_patterns:
            text = re.sub(pattern, '', text)

        text = text.strip()
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        return text
    
    def extract_device_name_from_chunks(self, chunks):
        for chunk in chunks:
            if ":" in chunk:
                device_part = chunk.split(":", 1)[0].strip()
                base_device = re.sub(r"\s*\(.*?\)", "", device_part).strip()
                if base_device:
                    return base_device
        return "the device"


def main():
    retriever = Retriever()
    # gen = Generatorr()

    retriever.read_and_chunk_csv("../data/final_dataset.csv")
    retriever.build_index()

    # user_query = "What are the reviews about Vivo Y91i?"
    user_query = "How is the battery life of Vivo Y91i?"

    results = retriever.search(user_query, top_k=5)

    print('\n🔎 Top Matching Results:')
    for res in results:
        print("-", res)

    device_name = retriever.extract_device_name_from_chunks(results)
    print('device_name is', device_name)

    # Combine reviews for summarization
    retrieved_passages = []
    for chunk in results:
        if ":" in chunk:
            _, review = chunk.split(":", 1)
        else:
            review = chunk
        retrieved_passages.append(review.strip())    

    # # Summarize with Ollaman qwen
    # summary = summarize_with_ollama(retrieved_passages, device_name, model='qwen2:1.5b')
    # print("\nSummary:")
    # print(summary)

    #  # Summarize with Ollama deepseek
    # summary_deepseek = summarize_with_deepseek(retrieved_passages, device_name)
    # print("------------------------------------------summary from deepseek------------------------------------------")
    # print(summary_deepseek)

    # Summarize with Ollama gemma
    summary_gemma = summarize_with_gemma(retrieved_passages, device_name)
    print("------------------------------------------summary from gemma------------------------------------------")
    print(summary_gemma)

    # # Summarize with Ollama_phi
    # summary_phi = summarize_with_phi(retrieved_passages, device_name)
    # print("------------------------------------------summary from phi------------------------------------------")
    # print(summary_phi)    
    
    


if __name__ == "__main__":
    main()
