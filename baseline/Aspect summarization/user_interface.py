import os
import json
import torch
import faiss
import time
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModel
import sys

# Ensure the parent directories are in the path for imports
sys.path.append(os.path.abspath(os.path.join("..", "generator")))
sys.path.append(os.path.abspath(os.path.join("..", "retriever")))

from generator import Generator
from retriever import Retriever

# Constants
CHUNK_FILE = os.path.join('..', 'data', "chunked_reviews.json")
INDEX_FILE = os.path.join('..', 'data', "reviews.index")

retriever = Retriever()
generator = Generator()
retrieved = [] # Global variable to hold retrieved chunks for the 'Show Reviews' button

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# The index is part of the retriever, so we don't need to load it here again.

# This function now takes the raw string input from Gradio and does everything.
def generate_summary_stream(user_query, aspects_str):
    global retrieved # Use the global variable to store results

    # 1. Handle empty inputs
    if not user_query:
        yield "Please enter a user query."
        return

    # 2. Parse the aspects string into a list
    aspects_list = [a.strip() for a in aspects_str.split(",") if a.strip()]
    if not aspects_list:
        yield "Please provide at least one aspect."
        return

    # 3. Retrieve relevant reviews
    retrieved = retriever.retrieve(user_query)

    if not retrieved:
        yield f"No reviews found for anything resembling: '{user_query}'"
        return

    # 4. Generate the summary using the generator module
    # The generator.generate_summary function returns the full summary string.
    summary = generator.generate_summary(user_query, retrieved, aspects_list)

    # 5. Yield the summary with a streaming effect
    output = ""
    for char in summary:
        output += char
        yield output
        time.sleep(0.02) # Keep the streaming feel


"""
Function to display retrieved chunks in a formatted JSON style
This function formats the retrieved chunks into a JSON-like structure for display.
"""
def display_chunks():
    global retrieved # Access the globally stored retrieved chunks
    if not retrieved:
        # Provide a default empty JSON structure if nothing is retrieved yet
        return json.dumps([], indent=2), gr.update(visible=True)

    json_output = json.dumps(retrieved, indent=2, ensure_ascii=False)
    return json_output, gr.update(visible=True)


""" Create the Gradio interface """
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
    min-width: 0 !important;
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

    query_input = gr.Textbox(label="User Query", placeholder="Ask about a product")
    output_text = gr.Textbox(show_label=False, lines=10, interactive=False) # Increased lines for better display
    chunks_output = gr.Code(language="json", visible=False, interactive=False)


    
    with gr.Row(elem_classes="centered-buttons"):
        generate_button = gr.Button("Summarize")
        show_chunks_button = gr.Button("Show Reviews")

    # Connect the button click directly to our main generator function.
    # Gradio will automatically handle the generator object and stream the output.
    generate_button.click(
        fn=generate_summary_stream, 
        inputs=[query_input], 
        outputs=output_text
    )
    
    # The 'Show Reviews' button logic is updated to use the global 'retrieved' variable
    show_chunks_button.click(
        fn=display_chunks, 
        inputs=[], 
        outputs=[chunks_output, chunks_output]
    )

demo.launch()
