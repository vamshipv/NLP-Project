import json
import logging
from datetime import datetime
import ollama
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from collections import defaultdict, Counter
import torch
import numpy as np
import sys
# import nltk

# Logging configuration
log_dir = os.path.join("..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "summary_log.json")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")

# Path to the chunked reviews file  os.path.join('..', 'data', 'reviews.json')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
chunked_file = os.path.join(project_root,"baseline", "data", "reviews.json")

# Get the directory of the current file (generator.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to baseline/ and into sentiment_analysis/
sentiment_path = os.path.join(current_dir, "..", "sentiment_analysis")
sentiment_path = os.path.abspath(sentiment_path)

sys.path.append(sentiment_path)

from sentiment_analyzer import SentimentAnalyzer

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
        self.sentiment_analyzer = SentimentAnalyzer()
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
    If an aspect is specified, it narrows the focus to that aspect only.
    It also includes sentiment analysis results to guide the summary generation.
    """

    def create_gemma_prompt(self, user_query, review_list, reviews_by_sentiment=None, aspect=None, sentiment_block=None):
        all_reviews_text = "\n".join(f"- {sentence}" for sentence in review_list)
        neg_pct = self.sentiment_analyzer.neg_count(reviews_by_sentiment, review_list)
        neg_instruction = ""
        if neg_pct < 20:
            neg_instruction = (
                "Since fewer than 20% of the reviews are negative, do not include negative feedback in the summary. "
            )

        if not aspect:
            return (
                f"Write a concise paragraph (6–7 sentences) summarizing customer reviews for '{user_query}'. "
                f"Use a neutral tone. Do not use bullet points or list pros and cons.\n\n"
                f"Also focus on common opinions with battery quality, build quality, performance and avoid mentioning specific reviews.\n\n"
                f"{neg_instruction}"
                f"The following sentiment analysis summarizes the tone of customer reviews:\n{sentiment_block}\n\n"
                f"Do not hallucinate or make up information. Just use the provided reviews.\n\n"
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
                f"{neg_instruction}"
                f"Only use the information provided in the reviews below. Do not make up or infer anything.\n\n"
                f"Here are the reviews:\n\n"
                f"{all_reviews_text}\n\n"
                f"Summary:"
            )
       
    """
    This method generates a summary of customer feedback based on the user query and the list of reviews.
    It uses the Gemma model via the Ollama API to create the summary and logs the process.
    It also performs sentiment analysis on the reviews to provide context for the summary.
    If no reviews are found, it returns a message indicating that.
    """
    def generate_summary(self, user_query, review_list, aspect=None):
        if not review_list:
            return "No relevant reviews found."
        
        sentiment_block, reviews_by_sentiment, aspect_scores = self.sentiment_analyzer.analyze_sentiment(review_list, aspect)

        prompt = self.create_gemma_prompt(user_query, review_list, reviews_by_sentiment, aspect, sentiment_block)

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": self.max_tokens,
                "temperature": 0.7}
        )

        final_summary = f"{response['message']['content']}\n\n{sentiment_block}"

        log_data = {
            "sentiment_analysis": sentiment_block,
            "model_prompt": prompt,
            "summary": final_summary
        }

        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
            f.write("\n")
        return final_summary, aspect_scores
    

if __name__ == "__main__":
    generator = Generator()
    
