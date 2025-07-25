import re
import sys
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import logging
from datetime import datetime

# Ensure the parent directories are in the path for imports
sys.path.append(os.path.abspath(os.path.join("..", "generator")))
sys.path.append(os.path.abspath(os.path.join("..", "retriever")))
sys.path.append(os.path.abspath(os.path.join("..", "product_matcher")))
title_path = os.path.join('..', 'data', 'brands.json')

# Load the product titles from the JSON file
with open(title_path, 'r', encoding='utf-8') as f:
    brands_data = json.load(f)
product_titles = [item.strip() for item in brands_data]

# Import the necessary classes from the generator and retriever modules
from generator import Generator
from retriever import Retriever
from product_matcher import ProductMatcher

nlp = spacy.load("en_core_web_sm")

# Logging configuration
log_dir = os.path.join("..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "summary_log.json")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")

"""
This module processes user queries to detect intent, clean the query, and generate a summary.
It acts as a bridge between the user interface and the generator/retriever modules.
It handles different scenarios such as bad words, gibberish, title-like queries, multiple titles, decision queries, and aspect-based queries.
It retrieves relevant chunks of reviews based on the cleaned query and generates a summary using the generator.
It logs the process, including the user query, number of chunks used, and the final summary.
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
        self.product_matcher = ProductMatcher()
        # This is used to detect title-like queries
        self.model_query = SentenceTransformer('all-MiniLM-L6-v2')
        self.aspect_keywords = {
            "battery": ["battery","charge", "mah", "power", "drain"],
            "camera": ["camera", "photo", "picture", "lens", "image", "zoom", "video"],
            "performance": ["smooth", "fast", "slow", "processor","performance"],
            "display": ["screen", "display", "brightness", "resolution", "refresh rate", "touch"],
            "build": ["build", "design", "material", "durability", "weight"],
            "software": ["ui", "os", "update", "bloatware", "interface", "android","software"],
            "heating": ["heating", "warm", "temperature", "overheat"]
        }

        self.aspects_keywords_not_avaliable = {
            "audio": ["audio", "sound", "speaker", "volume", "clarity", "bass", "mic", "microphone", "earpiece"],
            "price": ["price", "value", "worth", "expensive", "cheap", "budget", "overpriced", "cost"],
            "gaming": ["game", "fps", "graphics", "frame", "stutter", "heat during", "gaming"],
            "connectivity": ["wifi", "bluetooth", "network", "signal", "reception", "connectivity"],
            "storage": ["storage", "memory", "ram", "rom", "expandable", "sd card"],
            "security": ["fingerprint", "face unlock", "biometric", "sensor", "scanner", "unlock","security"],
            "accessories": ["charger", "case", "headphones", "earphones", "cable", "adapter", "accessory", "in-box", "accessories"],
            "charging_speed": ["charging speed", "fast charge", "wired", "wireless", "power delivery", "watt", "charge time", "charging_speed"]
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
    This method detects if the query mentions multiple product titles.
    It cleans the query using the product_matcher, extracts keywords using a keyword processor,
    and checks if the number of unique matches is greater than or equal to 2.
    """
    def detect_multiple_titles(self, query):
        cleaned_query = self.product_matcher.clean_query_for_brand_match(query)
        matches = self.product_matcher.keyword_processor.extract_keywords(cleaned_query)
        print(matches)
        return len(set(matches)) >= 2

    """
    This method extracts noun chunks from the query using spaCy's NLP model.
    It processes the query to identify noun phrases, which are potential candidates for aspect detection.
    """
    def extract_candidates(self, query):
        doc = nlp(query)
        return [chunk.text.lower() for chunk in doc.noun_chunks]
    
    """
    This method matches extracted candidates against predefined aspect categories.
    It checks each candidate against both supported and unsupported aspect keywords.
    It returns two lists: one for matched supported aspects and another for matched unsupported aspects.
    """
    def match_aspect_category(self, candidates):
        matched_supported = []
        matched_unsupported = []

        for phrase in candidates:
            print(phrase)
            for aspect, keywords in self.aspect_keywords.items():
                if any(keyword in phrase for keyword in keywords):
                    matched_supported.append(aspect)
            for aspect, keywords in self.aspects_keywords_not_avaliable.items():
                if any(keyword in phrase for keyword in keywords):
                    matched_unsupported.append(aspect)
    
        return list(set(matched_supported)), list(set(matched_unsupported))

    """
    This method detects a single valid aspect from the query.
    It extracts candidates from the query, matches them against aspect categories,
    and checks if the matched aspects are supported or unsupported.
    If an unsupported aspect is detected, it returns None and an error message.
    If multiple aspects are detected, it returns None and a message to focus on one aspect.
    If exactly one valid aspect is detected, it returns that aspect and None for the error message
    """
    def detect_single_valid_aspect(self, query):
        candidates = self.extract_candidates(query)
        matched_supported, matched_unsupported = self.match_aspect_category(candidates)

        if matched_unsupported:
            return None, "This aspect is not available for review summaries. Please try a different aspect or query."

        if len(matched_supported) == 0:
            return None, None

        if len(matched_supported) > 1:
            return None, "Multiple aspects detected. Please focus on one aspect at a time."

        return matched_supported[0], None
    
    """
    This method detects multiple aspects in the cleaned query.
    It checks if any of the predefined aspect keywords are present in the cleaned query.
    If no aspects are found, it returns None.
    If more than one aspect is found, it returns "multiple_aspects".
    If exactly one aspect is found, it returns that aspect.
    """
    def detect_multiple_aspects(self, cleaned_query):
        matched_aspects = []
        query_lower = cleaned_query.lower()

        for main_aspect, keywords in self.aspect_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                matched_aspects.append(main_aspect)
        
        if len(matched_aspects) == 0:
            return None  # no aspect found
        elif len(matched_aspects) > 1:
            return "multiple_aspects"  # more than one aspect found
        else:
            return matched_aspects[0]  # exactly one aspect found

    """
    This method filters sentences in the chunks based on the presence of keywords related to a specific aspect.
    It iterates through each chunk and checks if any of the keywords for the specified aspect are present in the chunk.
    It returns a list of chunks that contain at least one of the specified keywords.
    """
    def filter_sentences_by_aspect(self, chunks, keywords):
        return [chunk for chunk in chunks if any(keyword in chunk.lower() for keyword in keywords)]

    """
    This method checks if the query is a decision-making query.
    It looks for specific keywords that indicate a decision-making context, such as "buy", "purchase", or "recommend".
    It returns True if any of these keywords are found in the query, otherwise returns False.
    """
    def decision_query(self,query):
        decision_keywords = ["buy", "purchase", "recommend", "should I", "is it worth", "should I buy"]
        return any(keyword in query.lower() for keyword in decision_keywords)
    
    """
    This method checks if the text is gibberish by analyzing its content.
    It uses spaCy to process the text and checks if it contains at least three alphabetic
    tokens that are not out-of-vocabulary (OOV).
    If the text is too short or contains no valid alphabetic tokens, it is considered gibberish.
    """
    def _is_gibberish_check(self, text):
        MIN_CHAR_COUNT = 3
        if not text or len(text.strip()) < MIN_CHAR_COUNT:
            return True

        doc = nlp(text.lower())
        alpha_tokens = [token for token in doc if token.is_alpha]

        if not alpha_tokens:
            return True

        in_vocab_tokens = [token for token in alpha_tokens if not token.is_oov]
        return len(in_vocab_tokens) == 0 and len(alpha_tokens) < 3

    """
    This method cleans the user query by removing non-alphanumeric characters (except for hyphens and spaces),
    and normalizing whitespace to a single space. It also strips leading and trailing whitespace.
    """
    def clean_query(self, query):
        query = re.sub(r"[^\w\s\-]", "", query)
        return re.sub(r"\s+", " ", query).strip()

    """
    This method processes the user query to detect intent, clean the query, and generate a summary.
    It handles different scenarios such as bad words, gibberish, title-like queries, multiple titles,
    decision queries, and aspect-based queries.
    It retrieves relevant chunks of reviews based on the cleaned query and generates a summary using the generator.
    It logs the process, including the user query, number of chunks used, and the final summary.
    """
    def process(self, user_query):
        cleaned_query = self.clean_query(user_query)

        if not cleaned_query:
            return "Please enter a valid query.", "", ""

        if self.contains_bad_words(cleaned_query):
            return "Query contains inappropriate language. Please rephrase.", "", ""
        if self._is_gibberish_check(cleaned_query):
            return "Your query is not meaningful for system to understand. Please rephrase it.", "", ""
        
        if self.is_title_like_query(cleaned_query, self.product_titles):
            return "Please rephrase your query to focus on a product feedback by providing the correct product name.",  "", ""
        
        if self.detect_multiple_titles(cleaned_query):
            return "Your query seems to mention multiple products. Please focus on one product at a time.", "", ""

        if self.decision_query(cleaned_query):
            return "This system is designed to summarize product reviews, not to make purchase decisions. Please rephrase your query to focus on product feedback.", "", ""
        
        aspect, error = self.detect_single_valid_aspect(cleaned_query)
        if error:
            return error, "", ""\
            
        # aspect-based query
        if aspect:
            chunks = self.chunks_by_aspect(cleaned_query, aspect)
            if chunks is None or len(chunks) <= 4:
                return "Not enough reviews found for the specified aspect. Please try a different query.", "", ""
            
            log_data = {
            "Project": "Product Review Summarizer - Team Dave",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": cleaned_query,
            "reviews": [c["text"] if isinstance(c, dict) and "text" in c else c for c in chunks],
            "timestamp": datetime.now().isoformat(),
            "length_chunks": len(chunks),
            "Type_of_chunks": "aspect",
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("----------------------------------------LOG_START----------------------------------------\n")
                json.dump(log_data, f, indent=4)
                f.write("\n")

            filtered = self.filter_sentences_by_aspect(chunks, self.aspect_keywords[aspect])
            summary, scores = self.generator.generate_summary(cleaned_query, filtered, aspect=aspect)
            return summary, scores, chunks
        
        # for general query
        chunks = self.chunks_by_general(cleaned_query)
        if chunks is None or len(chunks) <= 4:
            return "Not enough reviews found for the specified aspect. Please try a different query.", "", ""
        log_data = {
            "Project": "Product Review Summarizer - Team Dave",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": cleaned_query,
            "reviews": [c["text"] if isinstance(c, dict) and "text" in c else c for c in chunks],
            "timestamp": datetime.now().isoformat(),
            "length_chunks": len(chunks),
            "Type_of_chunks": "general",
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("----------------------------------------LOG_START----------------------------------------\n")
            json.dump(log_data, f, indent=4)
            f.write("\n")
        
        summary, scores = self.generator.generate_summary(cleaned_query, chunks)
        return summary, scores, chunks

    """
    This method retrieves chunks of reviews based on the aspect specified in the query.
    It uses the retriever to get reviews related to the specified aspect.
    If no reviews are found, it returns a message indicating that no reviews were found for the query.
    """
    def chunks_by_aspect(self, query, aspect=None):
        retrieved_chunks_aspect = self.retriever.retrieve_by_aspect(query, aspect)
        if not retrieved_chunks_aspect:
            return None   
        return retrieved_chunks_aspect
    
    """
    This method retrieves chunks of reviews based on the general query.
    It uses the retriever to get reviews related to the query without focusing on a specific aspect.
    If no reviews are found, it returns a message indicating that no reviews were found for the query.
    """
    def chunks_by_general(self, query):
        retrieved_chunks_general = self.retriever.retrieve(query)
        if not retrieved_chunks_general:
            return None
        return retrieved_chunks_general
    
    """
    This method filters sentences in the chunks based on the presence of keywords related to a specific aspect.
    It iterates through each chunk and checks if any of the keywords for the specified aspect are present in the chunk.
    It returns a list of sentences that contain at least one of the specified keywords.
    """
    def filter_sentences_by_aspect(self, chunks, aspect_keywords):
        aspect_sentences = []
        for chunk in chunks:
            sentences = nltk.sent_tokenize(chunk["text"])
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in aspect_keywords):
                    aspect_sentences.append(sentence.strip())
        return aspect_sentences