import os
import json
import torch
import faiss
import time
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModel
import sys
from sentence_transformers import SentenceTransformer, util

# Ensure the parent directories are in the path for imports
sys.path.append(os.path.abspath(os.path.join("..", "generator")))
sys.path.append(os.path.abspath(os.path.join("..", "retriever")))
sys.path.append(os.path.abspath(os.path.join("..", "user_query_process")))

from generator import Generator
from retriever import Retriever
from user_query_process import User_query_process

# Constants
CHUNK_FILE = os.path.join('..', 'data', "chunked_reviews.json")
INDEX_FILE = os.path.join('..', 'data', "reviews.index")
title_path = os.path.join('..', 'data', 'brands.json')
with open(title_path, 'r', encoding='utf-8') as f:
    brands_data = json.load(f)

product_titles = [item.strip() for item in brands_data]


""" User Interface Class
This class provides a simple user interface for interacting with the product review summarization system.
It allows users to input queries, retrieve relevant chunks, and generate summaries using a pre-trained model.
It uses the Hugging Face Transformers library for model inference and FAISS for efficient similarity search.
It initializes the model and tokenizer, loads the chunked reviews and FAISS index,
and provides methods to get embeddings, retrieve relevant chunks, and generate summaries.
It is designed to be used in a Gradio web interface, allowing users to interact with the system easily."""
class user_interface:
    def __init__(self, chunk_file=CHUNK_FILE, faiss_index_file=INDEX_FILE):
        self.generator = Generator()
        self.retriever = Retriever()

        with open(chunk_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.index = faiss.read_index(faiss_index_file)

    # Below code is not used in the current implementation
    def get_embedding(self, text):
        inputs = self.tokenizer(f"query: {text}", return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0]
            return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")

    def retrieve(self, query, top_k=15):
        query_vec = self.get_embedding(query)
        _, indices = self.index.search(query_vec, top_k)
        return [self.chunks[i] for i in indices[0]]

# Initialize
user_interface = user_interface()
retriever = Retriever()
generator = Generator()
retrieved = []
query_processor = User_query_process()

""" 
Stream function to generate summaries based on user queries
This function retrieves relevant chunks based on the user query,
matches the product title, and generates a summary using the generator.

It yields the summary character by character to create a streaming effect.
It uses the retriever to find relevant chunks and the generator to create a summary.
It also handles cases where no reviews are found for the given query.

#TOOD
Need to improve the summary generation process to include sentiment analysis and key points and handling the case where no reviews are found.
"""

def generate_summary_stream(user_query):
    global retrieved
    if not user_query.strip():
        yield "Please enter a valid query."
        return

    try:
        retrieved = retriever.retrieve(user_query)
        result = query_processor.process(user_query)
        # print(f"Processed query: {user_query} -> Result: {result}")
        if isinstance(result, str):
            # It's a message, not a summary
            yield result
            return

        output = ""
        for char in result:
            output += char
            yield output
            time.sleep(0.02)

    except Exception as e:
        print(f"Error generating summary: {e}")
        yield "An error occurred while generating the summary. Please contact the developers."

""" 
Function to display retrieved chunks in a formatted JSON style
This function formats the retrieved chunks into a JSON-like structure for display.
"""
def display_chunks():
    if not retrieved:
        return "No chunks to display.", gr.update(visible=True)

    formatted_chunks = []
    for c in retrieved:
        chunk = (
            f"Model: {c.get('model', 'N/A')}\n"
            f"Brand: {c.get('brand', 'N/A')}\n"
            f"Stars: {c.get('stars', 'N/A')}\n"
            f"{c.get('text', '')}"
        )
        formatted_chunks.append(chunk)

    json_output = json.dumps(retrieved, indent=2, ensure_ascii=False)
    return json_output, gr.update(visible=True)


def contains_cuss_words(user_query):
    cuss_words = {
        "fuck", "shit", "bitch", "asshole", "bastard", "damn", "crap",
        "dick", "piss", "prick", "slut", "whore", "cunt"
    }
    words = user_query.lower().split()
    return any(word.strip('.,!?') in cuss_words for word in words)


""" Create the Gradio interface
This section sets up the Gradio interface with a clean and minimalist design.
It includes a title, input textbox for queries, buttons for generating summaries and displaying chunks,
and output textboxes for the summary and chunks.
"""

with gr.Blocks(css="""
body {
    background-color: white !important;
    margin: 0;
}
.gradio-container {
    background-color: white !important;
    font-family: 'Arial', sans-serif;
}

textarea, input {
    background-color: white !important;
    color: black !important;
    border: 1.5px solid black !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

textarea:focus, input:focus {
    outline: none !important;
    box-shadow: none !important;
    border: 1.5px solid black !important;
}

button {
    background-color: white !important;
    color: black !important;
    border: 1.5px solid black !important;
    border-radius: 16px !important;
    padding: 4px 10px !important;
    font-size: 12px !important;
    font-weight: 500;
    cursor: pointer;
    width: auto !important;
    min-width: 60px !important;
}

button:hover {
    background-color: #f5f5f5 !important;
}
               
.centered-buttons {
    justify-content: center !important;
    gap: 8px;
}

.centered-buttons button {
    font-size: 15px !important;
    padding: 2px 6px !important;
    width: 300px !important; 
    max-width: 300px !important;
    min-width: 0 !important;  /* prevent Gradio min-width overriding you */
    border-radius: 12px !important;
    text-align: center !important;
}
""") as demo:
    gr.Markdown(
        """
        <h1 style='text-align:left; font-weight:100; font-size:2.0em; color:black; margin-top: 10px; margin-bottom: 30px;'>
        Product Review Summarizer
        </h1>
        """
    )

    query_input = gr.Textbox(
        placeholder="Ask about a product",
        show_label=False,
        lines=2
    )

    with gr.Row(elem_classes="centered-buttons"):
        generate_button = gr.Button("Summarize")
        show_chunks_button = gr.Button("Show Reviews")

    summary_output = gr.Textbox(show_label=False, lines=6, interactive=False)
    chunks_output = gr.Code(language="json", visible=False, interactive=False)

    generate_button.click(generate_summary_stream, inputs=query_input, outputs=summary_output)
    show_chunks_button.click(display_chunks, inputs=[], outputs=[chunks_output, chunks_output])

demo.launch()
