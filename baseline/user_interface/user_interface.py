import os
import json
import torch
import faiss
import time
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModel
import sys

sys.path.append(os.path.abspath(os.path.join("..", "generator")))
sys.path.append(os.path.abspath(os.path.join("..", "retriever")))

from generator import Generator
from retriever import Retriever

# Constants
CHUNK_FILE = os.path.join('..', 'data', "chunked_reviews.json")
INDEX_FILE = os.path.join('..', 'data', "reviews.index")

class user_interface:
    def __init__(self, chunk_file=CHUNK_FILE, faiss_index_file=INDEX_FILE):
        self.generator = Generator()
        self.retriever = Retriever()

        with open(chunk_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.index = faiss.read_index(faiss_index_file)

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

def generate_summary_stream(user_query):
    global retrieved
    retrieved = retriever.retrieve(user_query)

    if not retrieved:
        yield f"No reviews found for anything resembling: '{user_query}'"
        return

    matched_product = f"{retrieved[0].get('brand', '')} {retrieved[0].get('model', '')}"
    yield f"Matched Product: {matched_product}\n\nGenerating summary..."

    summary = generator.generate_summary(user_query, retrieved)
    output = ""
    for char in summary:
        output += char
        yield output
        time.sleep(0.02)

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

# Clean minimalist design with compact buttons
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
