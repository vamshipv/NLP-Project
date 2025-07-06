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

# Path to the chunked reviews file
chunked_file = os.path.join('..', 'data', 'reviews.json')

class Generator:
    def __init__(self, chunk_file=chunked_file):
        """Initialize the Generator with the path to the chunked reviews file."""
        self.chunk_file = chunk_file
        self.max_tokens = 400
        self.model_name = "gemma2:2b"

        with open(self.chunk_file, "r", encoding="utf-8") as f:
            self.chunked_data = json.load(f)

    def create_gemma_prompt(self, user_query, review_list, aspects):
        """
        Creates a prompt for the Gemma model.
        - If 'aspects' are provided, it creates a targeted prompt.
        - Otherwise, it creates a general summary prompt.
        """
        all_reviews_text = "\n".join(f"- {chunk.get('text', '')}" for chunk in review_list if chunk.get('text'))
        print(aspects)
        if aspects:
            aspect_list_str = ", ".join(aspects)
            prompt = (
                f"You are a product review analyst. Based on the reviews for '{user_query}', "
                f"provide a detailed summary focusing ONLY on the following aspects: **{aspect_list_str}**. "
                f"For each aspect, clearly state the positive and negative points mentioned by customers. "
                f"If there is no information on an aspect, say so.\n\n"
                f"### Customer Reviews:\n{all_reviews_text}\n\n"
                f"### Aspect-Based Summary:"
            )
        else:
            prompt = (
                f"You are a product review analyst. Based on the reviews for '{user_query}', "
                f"provide a balanced and concise overall summary. "
                f"Highlight the main pros and cons mentioned by customers.\n\n"
                f"### Customer Reviews:\n{all_reviews_text}\n\n"
                f"### Overall Summary:"
            )
        
        return prompt

    def generate_summary(self, user_query, review_list, aspects):
        """
        Generates a summary. The 'aspects' parameter is a list of strings, or None/empty.
        """
        if not review_list:
            return "No relevant reviews found."
        aspects_to_summarize = ["battery", "camera"]

        # Query -> 
        # Get the aspect words from the query -> 
        # Match the main aspect word from the matched words -> 
        # Use a prompt accordingly
        for asp in user_query:
            if asp in aspects_to_summarize:
                prompt = self.create_gemma_prompt(user_query, review_list, aspects)
            else:
                prompt = self.create_gemma_prompt(user_query, review_list)

        if aspects is None:
            aspects = []

        # prompt = self.create_gemma_prompt(user_query, review_list, aspects)
        
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
            "aspects_queried": aspects,
            "num_chunks": len(review_list),
            "reviews": [c.get("text", "") for c in review_list],
            "prompt": prompt,
            "summary": final_summary
        }

        with open(log_path, "a", encoding="utf-8") as f:
            # THIS IS THE CORRECTED LINE
            json.dump(log_data, f, indent=4)
            f.write("\n")

        return final_summary


if __name__ == "__main__":
    # This block is for direct testing of the generator.py file
    print("--- Initializing Generator for Testing ---")
    generator = Generator()

    sample_reviews = [
        {'text': 'The battery life is amazing, lasts two full days! But the camera is a bit disappointing in low light.'},
        {'text': 'Great phone for the price. The display is vibrant and sharp.'},
        {'text': 'I hate the battery, it dies by lunchtime. Camera photos are surprisingly good though.'},
        {'text': 'Performance is smooth, no lag at all. Display is just okay, not very bright outdoors.'}
    ]

    # --- Test Case 1: Aspect-Based Summary ---
    user_query_1 = "Samsung Phone"
    aspects_to_summarize = ["battery", "camera"]
    print(f"\n--- Testing Aspect-Based Summary for '{user_query_1}' on {aspects_to_summarize} ---\n")
    summary1 = generator.generate_summary(user_query_1, sample_reviews, aspects=aspects_to_summarize)
    print("Generated Summary:\n", summary1)

    # --- Test Case 2: General Summary (no aspects) ---
    user_query_2 = "Samsung Phone"
    print(f"\n--- Testing General Summary for '{user_query_2}' (no aspects) ---\n")
    
         
