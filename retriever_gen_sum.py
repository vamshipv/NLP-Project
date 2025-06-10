from zoneinfo import ZoneInfo
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import os
import re
import multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

multiprocessing.set_start_method("fork", force=True)
import json
from datetime import datetime, UTC
from sentence_transformers import SentenceTransformer


class Generatorr:
    """
    Generator class with methods to build summarization prompt and generate answer.
    """
    # def __init__(self, model_name="facebook/bart-large-cnn", max_tokens=400):
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    #     self.max_tokens = max_tokens
    #     self.embedder = SentenceTransformer('all-MiniLM-L6-v2')




  

    def __init__(self):
        model_name = "knkarthick/MEETING_SUMMARY"
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize_reviews(self, reviews: list[str]) -> str:
        text = "\n".join(f"- {review}" for review in reviews)
        prompt = f"Summarize the following meeting transcript into 4-5 concise sentences:\n{text}"

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()