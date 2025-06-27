import json
import logging
from datetime import datetime
import ollama
import os
import re

# Logging configuration
log_dir = os.path.join("..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "summary_log.json")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")

# Path to the chunked reviews file
chunked_file = os.path.join('..', 'data', 'reviews.json')

class Generator:
    def __init__(self, chunk_file=chunked_file):
        """Initialize the Generator with the path to the chunked reviews file."""
        self.chunk_file = chunk_file
        self.max_tokens = 300
        self.model_name = "gemma2:2b"

        with open(self.chunk_file, "r", encoding="utf-8") as f:
            self.chunked_data = json.load(f)

    def create_gemma_prompt(self, user_query, review_list, aspects):
        aspect_list_str = ", ".join(aspects)
        # <<< CHANGE HERE: Use the correct key 'text' and use .get() for safety >>>
        all_reviews_text = "\n".join(f"- {chunk.get('text', '')}" for chunk in review_list)
        return (
            f"Summarize customer feedback for '{user_query}' focusing on these aspects: {aspect_list_str}. "
            f"Provide a concise summary of what customers are saying about each aspect, including both positive and negative feedback.\n\n"
            f"Reviews:\n{all_reviews_text}\n\n"
            "**Summary:**"
        )

    def generate_summary(self, user_query, review_list, aspects):
        if not review_list:
            return "No relevant reviews found."

        prompt = self.create_gemma_prompt(user_query, review_list, aspects)
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
            # <<< CHANGE HERE: Use the correct key 'text' for logging >>>
            "reviews": [c.get("text", "") for c in review_list],
            "prompt": prompt,
            "summary": final_summary
        }

        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
            f.write("\n")

        return final_summary

    def filter_reviews(self, user_query, reviews):
        """Filters reviews based on keywords in the user query."""
        keywords = ["battery", "display", "ui", "performance", "sound quality", "camera", "processor", "speed", "price"]
        relevant_reviews = []
        user_query_lower = user_query.lower()

        for review in reviews:
            # <<< CHANGE HERE: Use the correct key 'text' and .get() for safety >>>
            review_text_lower = review.get('text', '').lower() 
            
            if any(keyword in user_query_lower for keyword in keywords) or any(keyword in review_text_lower for keyword in keywords):
                relevant_reviews.append(review)
        return relevant_reviews

if __name__ == "__main__":
    generator = Generator()
    user_query = "Samsung Galaxy M51 battery and camera"
    aspects_to_summarize = ["battery", "camera"]

    relevant_reviews = generator.filter_reviews(user_query, generator.chunked_data)
    summary = generator.generate_summary(user_query, relevant_reviews, aspects_to_summarize)
    print(summary)