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

# Path to the chunked reviews file  os.path.join('..', 'data', 'reviews.json')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
chunked_file = os.path.join(project_root,"baseline", "data", "reviews.json")

"""
This generator module is designed to:
1. Load chunked reviews from a JSON file.
2. Generate a summary of customer feedback using the Gemma model.
3. Log the summary generation process, including the user query, number of chunks used, and the final summary.

It uses the Ollama API to interact with the Gemma model for text generation.
"""
class Generator:
    def __init__(self, chunk_file=chunked_file):
        """Initialize the Generator with the path to the chunked reviews file."""
        self.chunk_file = chunk_file
        self.max_tokens = 300
        self.model_name = "gemma2:2b"

        with open(self.chunk_file, "r", encoding="utf-8") as f:
            self.chunked_data = json.load(f)
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
    This method creates a prompt for the Gemma model to summarize customer feedback.
    It formats the user query and the list of reviews into a structured prompt.
    Currently, it uses a neutral tone and focuses on common opinions.

    #TODO
    # Work in progress to refine the prompt for better summarization and also to handle different tones or styles.
    Needs better summarization techniques to ensure the summary to inculde sentiment analysis and key points.
    """
    def create_gemma_prompt(self, user_query, review_list, aspect=None):
        all_reviews_text = "\n".join(f"- {sentence}" for sentence in review_list)
        if not aspect:
            return (
                f"Write a concise paragraph (6–7 sentences) summarizing customer reviews for the user query'{user_query}'. "
                f"Use a neutral tone. Do not use bullet points or list pros and cons.\n\n"
                f"Also Focus on common opinions with battery quality, build quality, performance and avoid mentioning specific reviews.\n\n"
                f"Do not hallucinate or make up information. Just used the provided reviews.\n\n"
                f"Here are the reviews:\n\n"
                f"{all_reviews_text}\n\n"
                f"Summary:"
            )
        else:
            return (
                f"Summarize the customer reviews for the user query: '{user_query}', focusing strictly on the aspect of '{aspect}'.\n"
                f"Do not include any information about other aspects such as "
                f"{', '.join(a for a in self.aspect_keywords if a != aspect)}.\n"
                f"Write a detailed paragraph (5–7 sentences) that reflects only the feedback related to '{aspect}'.\n"
                f"Only use the information provided in the reviews below. Do not make up or infer anything.\n\n"
                f"Here are the reviews:\n\n"
                f"{all_reviews_text}\n\n"
            )

    """
    This method generates a summary of customer feedback based on the user query and the list of reviews.
    It uses the Gemma model via the Ollama API to create the summary and logs the process.
    """
    def generate_summary(self, user_query, review_list, aspect=None):
        if not review_list:
            return "No relevant reviews found."

        prompt = self.create_gemma_prompt(user_query, review_list, aspect)
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
            "reviews": [c["text"] if isinstance(c, dict) and "text" in c else c for c in review_list],
            "prompt": prompt,
            "summary": final_summary
        }

        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
            f.write("\n")
        return final_summary


if __name__ == "__main__":
    generator = Generator()
    
