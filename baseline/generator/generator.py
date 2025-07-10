import json
import logging
from datetime import datetime
import ollama
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict, Counter
import torch
import numpy as np
import re
# import nltk

# nltk.download("vader_lexicon")

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

        # Initialize the Twitter RoBERTa sentiment model
        self.sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        self.aspect_keywords = {
            "battery": ["battery", "charge", "charging", "mah", "power", "drain"],
            "camera": ["camera", "photo", "picture", "lens", "image", "zoom", "video"],
            "performance": ["lag", "smooth", "fast", "slow", "processor", "snapdragon", "performance"],
            "display": ["screen", "display", "brightness", "resolution", "refresh rate", "touch"],
            "build": ["build", "design", "material", "durability", "weight", "feel"],
            "software": ["ui", "os", "update", "bloatware", "interface", "android", "software"],
            "heating": ["heat", "heating", "warm", "temperature", "overheat"]
        }

    def neg_count(self, reviews_by_sentiment):
        neg_count = len(reviews_by_sentiment.get("negative", [])) if reviews_by_sentiment else 0
        print('neg_count',neg_count)
        total_count = sum(len(v) for v in reviews_by_sentiment.values()) if reviews_by_sentiment else len(review_list)
        neg_pct = (neg_count / total_count) * 100 if total_count else 0
        return neg_pct

    """
    This method creates a prompt for the Gemma model to summarize customer feedback.
    It formats the user query and the list of reviews into a structured prompt.
    Currently, it uses a neutral tone and focuses on common opinions.

    #TODO
    # Work in progress to refine the prompt for better summarization and also to handle different tones or styles.
    Needs better summarization techniques to ensure the summary to inculde sentiment analysis and key points.
    """

    def create_gemma_prompt(self, user_query, review_list, reviews_by_sentiment=None, aspect=None, sentiment_block=None):
        all_reviews_text = "\n".join(f"- {sentence}" for sentence in review_list)
        neg_pct = self.neg_count(reviews_by_sentiment)
        neg_instruction = "" # should we add something in the default ? #TODO
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
    """
    def generate_summary(self, user_query, review_list, aspect=None):
        if not review_list:
            return "No relevant reviews found."
        
        sentiment_block, reviews_by_sentiment, aspect_scores = self.analyze_sentiment(review_list, aspect)

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
            "Project": "Product Review Summarizer - Team Dave",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": user_query,
            "reviews": [c["text"] if isinstance(c, dict) and "text" in c else c for c in review_list],
            "sentiment_analysis": sentiment_block,
            "aspect": aspect if aspect else "general",
            "model_prompt": prompt,
            "summary": final_summary
        }

        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
            f.write("\n")
        return final_summary, aspect_scores
    
    """
    This method performs sentiment analysis on a list of reviews using a transformer-based 
    sentiment classification model. It optionally filters sentiment analysis by a specific aspect 
    (e.g., battery, camera). If no aspect is provided, it maps sentiments to detected aspects 
    using predefined keywords.      
    """
    def analyze_sentiment(self, review_list, aspect=None):
        sentiment_analyzer = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)
        aspects = list(self.aspect_keywords.keys())
        aspect_sentiments = defaultdict(list) 
        general_counts = Counter()
        reviews_by_sentiment = {"positive": [], "negative": [], "neutral": []}

        for review in review_list:
            review_text = review["text"] if isinstance(review, dict) and "text" in review else review
            review_text_lower = review_text.lower()
            # tokenize review into words (simple split on non-word chars)
            tokens = set(re.findall(r'\b\w+\b', review_text_lower))

            result = sentiment_analyzer(review_text)[0]
            label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
            label = label_map.get(result["label"], "neutral")

            reviews_by_sentiment[label].append(review_text)

            if aspect:
                if any(kw in tokens for kw in self.aspect_keywords.get(aspect, [])):
                    general_counts[label] += 1
            else:
                matched = False
                for asp, keywords in self.aspect_keywords.items():
                    if any(kw in tokens for kw in keywords):
                        aspect_sentiments[label].append(asp)
                        matched = True
                        break
                if not matched:
                    aspect_sentiments[label].append("general")

        def calculate_percentage_breakdown(pos, neg, neu):
            total = pos + neg + neu
            if total == 0:
                return {"positive": 0, "negative": 0, "neutral": 0}

            raw = {
                "positive": (pos / total) * 100,
                "negative": (neg / total) * 100,
                "neutral":  (neu / total) * 100
            }

            rounded = {k: round(v) for k, v in raw.items()}
            diff = 100 - sum(rounded.values())

            # Fix rounding difference
            if diff != 0:
                # Sort by decimal difference to adjust the closest
                adjust_order = sorted(raw, key=lambda k: raw[k] - rounded[k], reverse=(diff > 0))
                for i in range(abs(diff)):
                    rounded[adjust_order[i % 3]] += 1 if diff > 0 else -1

            return rounded
        
        total_reviews = len(review_list)

        if aspect:
            pos = general_counts["positive"]
            neg = general_counts["negative"]
            neu = general_counts["neutral"]

            percentages = calculate_percentage_breakdown(pos, neg, neu)
            overall = max(percentages, key=percentages.get).capitalize()

            pos = [aspect] if general_counts["positive"] > 0 else []
            neg = [aspect] if general_counts["negative"] > 0 else []
            neu = [aspect] if general_counts["neutral"] > 0 else []
            aspect_scores = self.convert_to_aspect_scores(percentages, pos, neg, neu)
            sentiment_block = (
                f"Overall Sentiment : {overall}\n"
                f"- {percentages['positive']}% of customers have positive reviews.\n"
                f"- {percentages['negative']}% of customers have negative reviews.\n"
                f"- {percentages['neutral']}% of customers have neutral reviews.\n\n"
            )
        else:
            pos_aspects = [a for a in aspect_sentiments["positive"] if a != "general"]
            neg_aspects = [a for a in aspect_sentiments["negative"] if a != "general"]
            neu_aspects = [a for a in aspect_sentiments["neutral"] if a != "general"]

            pos = len(aspect_sentiments["positive"])
            neg = len(aspect_sentiments["negative"])
            neu = len(aspect_sentiments["neutral"])

            percentages = calculate_percentage_breakdown(pos, neg, neu)
            overall = max(percentages, key=percentages.get).capitalize()
            pos = [aspect] if general_counts["positive"] > 0 else []
            neg = [aspect] if general_counts["negative"] > 0 else []
            neu = [aspect] if general_counts["neutral"] > 0 else []
            aspect_scores = self.convert_to_aspect_scores(percentages, pos_aspects, neg_aspects, neu_aspects)
            sentiment_block = (
                f"OVERALL SENTIMENT : {overall}\n"
                f"- {percentages['positive']}% of customers have positive reviews about {', '.join(sorted(set(pos_aspects))) or 'various aspects'}.\n"
                f"- {percentages['negative']}% have negative reviews about {', '.join(sorted(set(neg_aspects))) or 'various aspects'}.\n"
                f"- {percentages['neutral']}% have neutral reviews about {', '.join(sorted(set(neu_aspects))) or 'various aspects'}.\n\n"
            )
        print("Sentiment Block:", sentiment_block)
        print("Reviews by Sentiment:", reviews_by_sentiment)
        print("Aspect Scores:", aspect_scores)
        
        return sentiment_block, reviews_by_sentiment, aspect_scores

    def convert_to_aspect_scores(self, percentages, pos_aspects, neg_aspects, neu_aspects):
        aspect_scores = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
        for aspect in pos_aspects:
            aspect_scores[aspect]["positive"] = percentages["positive"]
        for aspect in neg_aspects:
            aspect_scores[aspect]["negative"] = percentages["negative"]
        for aspect in neu_aspects:
            aspect_scores[aspect]["neutral"] = percentages["neutral"]
        
        aspect_scores = json.dumps(aspect_scores, indent=2)
        print("Aspect Scores:", aspect_scores)
        return aspect_scores


if __name__ == "__main__":
    generator = Generator()
    
