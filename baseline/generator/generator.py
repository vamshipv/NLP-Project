import json
import logging
from datetime import datetime
import ollama
import os

# Logging configuration
log_dir = os.path.join("..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "summary_log.json")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")
chunked_file = os.path.join('..', 'data', 'reviews.json')

class Generator:
    def __init__(self, chunk_file=chunked_file):
        self.chunk_file = chunk_file
        self.max_tokens = 300
        self.model_name = "gemma2:2b"

        with open(self.chunk_file, "r", encoding="utf-8") as f:
            self.chunked_data = json.load(f)

    def create_gemma_prompt(self, user_query, review_list):
        all_reviews_text = "\n".join(f"- {chunk['text']}" for chunk in review_list)
        return (
            f"Summarize customer feedback for '{user_query}' in a neutral tone using the reviews below. "
            f"Keep the summary concise and focus only on common opinions, concerns, or recurring sentiments.\n\n"
            f"{all_reviews_text}\n\n"
            "**Summary:**"
        )

    def generate_summary(self, user_query, review_list):
        if not review_list:
            return "No relevant reviews found."

        prompt = self.create_gemma_prompt(user_query, review_list)
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": self.max_tokens}
        )

        final_summary = response["message"]["content"]

        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model_name,
            "user_query": user_query,
            "num_chunks": len(review_list),
            "reviews": [c["text"] for c in review_list],
            "prompt": prompt,
            "summary": final_summary
        }

        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
            f.write("\n")

        return final_summary


if __name__ == "__main__":
    generator = Generator()
    
