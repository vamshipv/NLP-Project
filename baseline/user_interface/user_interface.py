from collections import defaultdict
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
import re
import logging

# Ensure the parent directories are in the path for imports
sys.path.append(os.path.abspath(os.path.join("..", "user_query_process")))

from user_query_process import User_query_process


# Logging configuration
log_dir = os.path.join("..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "summary_log.json")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")

"""This module provides a user interface for generating product review summaries.
It uses Gradio to create a web interface where users can input queries about products.
The interface allows users to:
1. Enter a query about a product.
2. Generate a summary of customer feedback based on the query.
3. Display relevant chunks of reviews in a formatted JSON style.
It also includes functionality to handle sentiment analysis and display the results in a user-friendly manner.

The user interface is designed to be clean and minimalist, focusing on usability and accessibility.
It includes features such as:
- A title for the application.
- An input textbox for user queries.
- Buttons for generating summaries and displaying chunks.
- Output textboxes for displaying the summary and sentiment analysis.
"""
class user_interface:
    def __init__(self):
        self.retrieved = []
        self.query_processor = User_query_process()
        self.retrieved_chunks_for_display = []

    """
    Function to generate a summary stream based on user query
    This function processes the user query, retrieves relevant chunks, and generates a summary.
    It yields the summary text and sentiment analysis results in a streaming manner.
    It handles errors gracefully and provides feedback to the user if the query is invalid or if an error occurs during processing.
    """
    def generate_summary_stream(self, user_query):

        if not user_query.strip():
            yield "", "<p>Please enter a valid query.</p>"
            return
        # retrieved = query_processor.check_chunks(user_query)
        try:
            
            summary_text, aspect_score, retrieved_chunks = self.query_processor.process(user_query)
            self.retrieved_chunks_for_display = retrieved_chunks
            log_data = {
            }

            with open(log_path, "a", encoding="utf-8") as f:
                json.dump(log_data, f, indent=4)
                f.write("\n----------------------------------------LOG_END----------------------------------------n")
                f.write("\n")

            if isinstance(aspect_score, str):
                try:
                    aspect_score = json.loads(aspect_score)
                except json.JSONDecodeError:
                    aspect_score = {}

            sentiment_html = self.render_all_bars(aspect_score) if isinstance(aspect_score, dict) else "<p>Invalid sentiment data.</p>"

            yield "", sentiment_html

            tokens = re.split(r'(\n+|\s+)', summary_text)

            tokens = [token for token in tokens if token]

            output = ""
            for i, token in enumerate(tokens):
                output += token
                yield output, sentiment_html
                if '\n' in token:
                    time.sleep(0.2) 
                else:
                    time.sleep(0.05) 

            

            yield summary_text, sentiment_html 

        except Exception as e:
            print(f"Error generating summary: {e}")
            yield "An error occurred while generating the summary. Please contact the developers.", ""

    """
    Function to render sentiment bar for each aspect
    This function creates an HTML representation of a sentiment bar for a given aspect.
    It takes the aspect name and its sentiment scores (positive, neutral, negative) as input.
    The sentiment scores are used to create a visual representation of the sentiment distribution.
    The function returns a string containing the HTML code for the sentiment bar.
    """
    def render_sentiment_bar(self, aspect, scores):
        pos = scores.get("positive", 0)
        neu = scores.get("neutral", 0)
        neg = scores.get("negative", 0)

        return f"""
        <div style="padding: 12px; margin-bottom: 12px; font-size: 13px; font-family: 'Helvetica Neue', sans-serif; color: #111;
                    border: 1px solid #ddd; border-radius: 8px; background-color: #fff; min-width: 260px; flex: 1;">
            <div style="margin-bottom: 8px; font-weight: 600; text-align: center;">{aspect.capitalize()}</div>
            <div style="display: flex; flex-direction: row; align-items: center; gap: 8px;"> <div style="height: 10px; border-radius: 5px; overflow: hidden; display: flex; flex-grow: 1; background-color: #e0e0e0; box-shadow: inset 0 1px 2px rgba(0,0,0,0.08);">
                    <div style="width: {pos}%; background-color: #222;"></div>
                    <div style="width: {neu}%; background-color: #999;"></div>
                    <div style="width: {neg}%; background-color: #666;"></div>
                </div>
                <div style="font-size: 12px; color: #333; white-space: nowrap;"> {pos}% / {neu}% / {neg}%
                </div>
            </div>
        </div>
        """

    """
    Function to render all sentiment bars
    This function takes a dictionary of sentiment scores for different aspects and generates HTML for all sentiment bars.
    It iterates through the dictionary, calling `render_sentiment_bar` for each aspect and its corresponding scores.
    The resulting HTML strings are joined together to create a single HTML block that contains all sentiment bars.
    The function returns a string containing the complete HTML for all sentiment bars.
    """
    def render_all_bars(self, sentiment_dict):
        all_bars_html = "\n".join(self.render_sentiment_bar(a, s) for a, s in sentiment_dict.items())
        return f"""
        <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
            {all_bars_html}
        </div>
        """

    """ 
    Function to display retrieved chunks in a formatted JSON style
    This function formats the retrieved chunks into a JSON-like structure for display.
    """
    def display_chunks(self):
        if not self.retrieved_chunks_for_display:
            return "No chunks to display.", gr.update(visible=True)
        
        formatted_chunks = []
        for c in self.retrieved_chunks_for_display:
            chunk = (
                f"Model: {c.get('model', 'N/A')}\n"
                f"Brand: {c.get('brand', 'N/A')}\n"
                f"Stars: {c.get('stars', 'N/A')}\n"
                f"Aspect: {c.get('aspect', 'N/A')}\n"
                f"{c.get('text', '')}"
            )
            formatted_chunks.append(chunk)

        json_output = json.dumps(self.retrieved_chunks_for_display, indent=2, ensure_ascii=False)
        return json_output, gr.update(visible=True)


    """
    Function to launch the Gradio interface
    This function sets up the Gradio Blocks interface with the necessary components.
    It includes a title, input textbox for user queries, buttons for generating summaries and displaying chunks,
    and output textboxes for displaying the summary and sentiment analysis.
    The interface is styled with custom CSS to ensure a clean and minimalist design.
    The function uses Gradio's Blocks API to create a responsive layout with rows and columns.
    The `launch_interface` method is called to start the Gradio app.
    """
    def launch_interface(self):
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
            min-width: 0 !important;   /* prevent Gradio min-width overriding you */
            border-radius: 12px !important;
            text-align: center !important;
        }

        /* Styles for the sentiment bars container */
        .sentiment-bars-container {
            display: flex;
            flex-wrap: wrap; /* Allows items to wrap to the next line */
            gap: 10px; /* Space between the sentiment bars */
            justify-content: center; /* Center the bars horizontally */
            margin-top: 15px; /* Add some space above the bars */
        }

        /* Style for individual sentiment bar containers to ensure they align */
        .sentiment-bar-box {
            padding: 12px;
            font-size: 13px;
            font-family: 'Helvetica Neue', sans-serif;
            color: #111;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fff;
            min-width: 260px; /* Adjust as needed */
            flex: 1; /* Allows the bars to grow and shrink */
            box-sizing: border-box; /* Include padding and border in the element's total width and height */
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
                show_chunks_button = gr.Button("Show Chunks")

            with gr.Row():
                with gr.Column(scale=4):
                    summary_output = gr.Textbox(show_label=False, lines=6, interactive=False)
            with gr.Row():
                with gr.Column(scale=1):
                    sentiment_display = gr.HTML(label="Sentiment Breakdown", elem_classes="sentiment-bars-container")

            chunks_output = gr.Code(language="json", visible=False, interactive=False)
            generate_button.click(self.generate_summary_stream, inputs=query_input, outputs=[summary_output, sentiment_display])
            show_chunks_button.click(self.display_chunks, inputs=[], outputs=[chunks_output, chunks_output])

        demo.launch()

if __name__ == "__main__":
    user_interface = user_interface()
    user_interface.launch_interface()
