# baseline/generator/generator.py

from typing import List
from transformers import pipeline

from transformers.utils import logging

# Suppress model loading logs
logging.set_verbosity_error()

class Generator:
    def __init__(self, hf_model: str = "sshleifer/distilbart-cnn-12-6"):
        """
        Initialize the summarization model from Hugging Face.
        """
        self.summarizer = pipeline("summarization", model=hf_model)

    def build_prompt(self, chunks: List[str], question: str) -> str:
        """
        Combine retrieved text chunks into a single input for summarization.
        The question isn't explicitly used here but could be added for context.
        """
        context = "\n\n".join(chunks)
        return context

    def generate_answer(self, prompt: str) -> str:
        """
        Generate a summary using the model. Assumes `prompt` is the full context.
        """
        result = self.summarizer(prompt, max_length=150, min_length=40, do_sample=False)
        return result[0]['summary_text']
