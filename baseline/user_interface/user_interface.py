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

# Initialize retriever and generator
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
            f"Review: {c.get('text', '')}"
        )
        formatted_chunks.append(chunk)

    return "\n\n---\n\n".join(formatted_chunks), gr.update(visible=True)

# Build UI
with gr.Blocks() as demo:
    gr.Markdown("## 💬 Product Review Summarizer (Powered by Gemma + E5)")
    query_input = gr.Textbox(label="Your Product Query", placeholder="e.g. Battery life of Realme Narzo 20", lines=2)

    generate_button = gr.Button("Generate Summary")
    summary_output = gr.Textbox(label="Generated Summary", lines=6, visible=True)

    show_chunks_button = gr.Button("Show Retrieved Review Chunks")
    chunks_output = gr.Textbox(label="Top Review Chunks", lines=10, visible=False)

    generate_button.click(generate_summary_stream, inputs=query_input, outputs=summary_output)
    show_chunks_button.click(display_chunks, inputs=[], outputs=[chunks_output, chunks_output])

demo.launch()
