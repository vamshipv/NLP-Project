import re
import sys
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize

sys.path.append(os.path.abspath(os.path.join("..", "generator")))
sys.path.append(os.path.abspath(os.path.join("..", "retriever")))
title_path = os.path.join('..', 'data', 'brands.json')

with open(title_path, 'r', encoding='utf-8') as f:
    brands_data = json.load(f)

product_titles = [item.strip() for item in brands_data]

from generator import Generator
from retriever import Retriever

"""
TODO Work in progress
This module processes user queries to detect intent, clean the query, and generate summaries based on retrieved product reviews.
It uses a retriever to fetch relevant chunks of reviews and a generator to create summaries.
It also includes functionality to detect aspects of the query and handle inappropriate language.
"""
class User_query_process:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.intent = None
        self.sentiment = None
        self.product_titles = [item.strip() for item in brands_data]
        self.bad_words = {
        "fuck", "shit", "bitch", "asshole", "bastard", "damn", "crap",
        "dick", "piss", "prick", "slut", "whore", "cunt"
    }
        self.retrieved_chunks_aspect = []
        self.retrieved_chunks_general = []

        # This is used to detect title-like queries
        self.model_query = SentenceTransformer('all-MiniLM-L6-v2')
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
    This method checks if the query contains any bad words.
    It splits the query into words, normalizes them to lowercase, and checks against a set of predefined bad words.
    """
    def contains_bad_words(self, query):
        words = set(query.lower().split())
        return any(word.strip('.,!?') in self.bad_words for word in words)

    """    
    This method checks if the query is similar to any of the product titles.
    It encodes the query and the product titles into embeddings using a SentenceTransformer model,
    calculates the cosine similarity between the query embedding and each title embedding,
    and returns True if the maximum similarity score exceeds a specified threshold (default is 0.90).
    """
    def is_title_like_query(self, query, titles, threshold=0.90):
        query_emb = self.model_query.encode(query, convert_to_tensor=True)
        title_embs = self.model_query.encode(titles, convert_to_tensor=True)
        sim_scores = util.cos_sim(query_emb, title_embs)[0]
        return np.max(sim_scores.numpy()) >= threshold

    """
    This method detects the intent of the user query.
    It checks if the query is similar to product titles, contains aspect keywords, or is a decision-making query.
    It returns a string indicating the detected intent: "title_query", "aspect", "decision_query", or "summary".
    """
    def detect_intent(self, query):
        q = query.lower()
        if (self.is_title_like_query(q, self.product_titles)):
            return "title_query"

        if any(word in q for word in self.aspect_keywords):
            print(f"Detected aspect in query: {q}")
            return "aspect"

        if any(phrase in q for phrase in [
            "should i buy", "is this product good", "what should i buy",
            "is it worth it", "which one is better", "do you recommend"
        ]):
            return "decision_query"

        return "summary"

    """
    This method cleans the user query by removing non-alphanumeric characters (except for hyphens and spaces),
    and normalizing whitespace to a single space. It also strips leading and trailing whitespace.
    """
    def clean_query(self, query):
        query = re.sub(r"[^\w\s\-]", "", query)
        return re.sub(r"\s+", " ", query).strip()

    """
    TODO Work in progress
    This method processes the user query to detect intent, clean the query, and generate a summary.
    It handles different intents such as title queries, decision queries, and aspect-based queries.
    It retrieves relevant chunks of reviews based on the cleaned query and generates a summary using the generator.
    If the query is empty or contains inappropriate language, it returns appropriate messages."""
    def process(self, user_query):
        if self.intent == "title_query":
            return "Please rephrase your query to focus on a product feedback by providing the correct product name."
        
        if not user_query.strip():
            return "Please enter a valid query."

        if self.contains_bad_words(user_query):
            return "Query contains inappropriate language. Please rephrase."

        self.intent = self.detect_intent(user_query)
        cleaned_query = self.clean_query(user_query)

        if self.intent == "decision_query":
            return "This system is designed to summarize product reviews, not to make purchase decisions. Please rephrase your query to focus on product feedback."

        if self.intent == "aspect":
            for aspect in self.aspect_keywords:
                if aspect in cleaned_query.lower():
                    retrieved_chunks_aspect = self.chunks_by_aspect(user_query, aspect=aspect)
                    if len(retrieved_chunks_aspect) <= 4:
                        return "Not enough reviews found for the specified aspect. Please try a different query."
                    print(f"Retrieved chunks for aspect '{aspect}': {retrieved_chunks_aspect}")
                    retrieved_chunks_aspect = self.filter_sentences_by_aspect(retrieved_chunks_aspect, aspect)
                    summary = self.generator.generate_summary(user_query, retrieved_chunks_aspect, aspect=aspect)
                    return summary 
                
        retrieved_chunks_general = self.chunks_by_general(user_query)
        if len(retrieved_chunks_general) <= 4:
            return "Not enough reviews found for the specified aspect. Please try a different query."
        print(f"Retrieved chunks general : {retrieved_chunks_general}")
        summary = self.generator.generate_summary(user_query, retrieved_chunks_general)
        return summary

    def chunks_by_aspect(self, query, aspect=None):
        retrieved_chunks_aspect = self.retriever.retrieve_by_aspect(query, aspect)
        if not retrieved_chunks_aspect:
            return "No reviews found for your query."   
        return retrieved_chunks_aspect
    
    def chunks_by_general(self, query):
        retrieved_chunks_general = self.retriever.retrieve(query)
        if not retrieved_chunks_general:
            return "No reviews found for your query."
        return retrieved_chunks_general
    
    def check_chunks(self, query):
        if self.detect_intent(query) == "aspect":
            for aspect in self.aspect_keywords:
                if aspect in self.clean_query(query):
                    retrieved_chunks_aspect = self.chunks_by_aspect(query, aspect=aspect)
                    return retrieved_chunks_aspect
        retrieved_chunks_general = self.chunks_by_general(query)
        return retrieved_chunks_general
    
    def filter_sentences_by_aspect(self, chunks, aspect_keywords):
        aspect_sentences = []
        for chunk in chunks:
            sentences = nltk.sent_tokenize(chunk["text"])
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in aspect_keywords):
                    aspect_sentences.append(sentence.strip())
        return aspect_sentences