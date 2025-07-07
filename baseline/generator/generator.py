import json
import logging
from datetime import datetime
import ollama
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Logging configuration
log_dir = os.path.join("..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "summary_log.json")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")

# Path to the chunked reviews file  os.path.join('..', 'data', 'reviews.json')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
chunked_file = os.path.join(project_root,"baseline", "data", "reviews.json")

# # Load sentiment model
# sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
# sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
# sentiment_labels = ["negative", "neutral", "positive"]

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

        # Sentiment classifier model (Hugging Face Transformers)
        self.classifier_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.sentiment_labels = ["negative", "neutral", "positive"]

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.classifier_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.classifier_model_name)


        with open(self.chunk_file, "r", encoding="utf-8") as f:
            self.chunked_data = json.load(f)

    """
    This method creates a prompt for the Gemma model to summarize customer feedback.
    It formats the user query and the list of reviews into a structured prompt.
    Currently, it uses a neutral tone and focuses on common opinions.

    #TODO
    # Work in progress to refine the prompt for better summarization and also to handle different tones or styles.
    Needs better summarization techniques to ensure the summary to inculde sentiment analysis and key points.
    """
    def create_gemma_prompt(self, user_query, review_list):
        # print(review_list)
        all_reviews_text = "\n".join(f"- {chunk['text']}" for chunk in review_list)
        # print(all_reviews_text)
        return (
            f"Write a concise paragraph (6–7 sentences) summarizing customer reviews for '{user_query}'. "
            f"Use a neutral tone. Do not use bullet points or list pros and cons.\n\n"
            f"Also Focus on common opinions with battery quality, build quality, performance and avoid mentioning specific reviews.\n\n"
            f"Do not hallucinate or make up information. Just used the provided reviews.\n\n"
            f"Here are the reviews:\n\n"
            f"{all_reviews_text}\n\n"
            f"Summary:"
        )
    
    def classify_sentiment(self, text):
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.sentiment_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        label_id = torch.argmax(probs, dim=-1).item()
        return self.sentiment_labels[label_id]


    """
    This method generates a summary of customer feedback based on the user query and the list of reviews.
    It uses the Gemma model via the Ollama API to create the summary and logs the process.
    """
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
        sentiment = self.classify_sentiment(final_summary)
        print('-------sentiment-----',sentiment)


        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model_name,
            "user_query": user_query,
            "num_chunks": len(review_list),
            "reviews": [c["text"] for c in review_list],
            "prompt": prompt,
            "summary": final_summary,
            "sentiment": sentiment
        }

        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
            f.write("\n")

        return final_summary, sentiment


if __name__ == "__main__":
    generator = Generator()
    
