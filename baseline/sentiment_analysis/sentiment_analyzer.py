import re
from collections import defaultdict, Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json

class SentimentAnalyzer:
    def __init__(self):
        """
        Initializes the SentimentAnalyzer with a pre-trained sentiment model.
        """
        self.sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
        # Use MPS if available, otherwise CUDA, otherwise CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=self.device.index if self.device.type != 'cpu' else -1)

        self.aspect_keywords = {
            "battery": ["battery", "charge", "charging", "mah", "power", "drain"],
            "camera": ["camera", "photo", "picture", "lens", "image", "zoom", "video"],
            "performance": ["lag", "smooth", "fast", "slow", "processor", "snapdragon", "performance"],
            "display": ["screen", "display", "brightness", "resolution", "refresh rate", "touch"],
            "build": ["build", "design", "material", "durability", "weight", "feel"],
            "software": ["ui", "os", "update", "bloatware", "interface", "android", "software"],
            "heating": ["heat", "heating", "warm", "temperature", "overheat"]
        }

    def neg_count(self, reviews_by_sentiment, review_list):
        neg_count = len(reviews_by_sentiment.get("negative", [])) if reviews_by_sentiment else 0
        # print('neg_count',neg_count)
        total_count = sum(len(v) for v in reviews_by_sentiment.values()) if reviews_by_sentiment else len(review_list)
        neg_pct = (neg_count / total_count) * 100 if total_count else 0
        return neg_pct
    
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
        total_reviews = len(review_list)

        if aspect:
            pos = general_counts["positive"]
            neg = general_counts["negative"]
            neu = general_counts["neutral"]

            percentages = self.calculate_percentage_breakdown(pos, neg, neu)
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

            percentages = self.calculate_percentage_breakdown(pos, neg, neu)
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
        # print("Sentiment Block:", sentiment_block)
        # print("Reviews by Sentiment:", reviews_by_sentiment)
        # print("Aspect Scores:", aspect_scores)
        
        return sentiment_block, reviews_by_sentiment, aspect_scores

    def calculate_percentage_breakdown(self, pos, neg, neu):
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

    def convert_to_aspect_scores(self, percentages, pos_aspects, neg_aspects, neu_aspects):
        aspect_scores = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
        for aspect in pos_aspects:
            aspect_scores[aspect]["positive"] = percentages["positive"]
        for aspect in neg_aspects:
            aspect_scores[aspect]["negative"] = percentages["negative"]
        for aspect in neu_aspects:
            aspect_scores[aspect]["neutral"] = percentages["neutral"]
        
        aspect_scores = json.dumps(aspect_scores, indent=2)
        # print("Aspect Scores:", aspect_scores)
        return aspect_scores
