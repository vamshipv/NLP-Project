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
            yield "", "<p>Please enter a valid query.</p>", gr.update(visible=True)
            return

        try:

            summary_text, aspect_score, retrieved_chunks = self.query_processor.process(user_query)
            self.retrieved_chunks_for_display = retrieved_chunks

            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n----------------------------------------LOG_END------------------------------------------")
                f.write("\n")

            if isinstance(aspect_score, str):
                try:
                    aspect_score = json.loads(aspect_score)
                except json.JSONDecodeError:
                    aspect_score = {}

            sentiment_html = self.render_all_bars(aspect_score) if isinstance(aspect_score, dict) else "<p>Invalid sentiment data.</p>"

            yield "", sentiment_html, gr.update(visible=True)

            tokens = re.split(r'(\n+|\s+)', summary_text)
            tokens = [token for token in tokens if token]

            output = ""
            for i, token in enumerate(tokens):
                output += token
                yield output, sentiment_html, gr.update(visible=True)
                if '\n' in token:
                    time.sleep(0.2)
                else:
                    time.sleep(0.05)

            yield summary_text, sentiment_html, gr.update(visible=True)

        except Exception as e:
            yield "An error occurred while generating the summary. Please check if ollama is running in the background or contact the developers.", ""


    def render_sentiment_bar(self, aspect, scores):
        pos = scores.get("positive", 0)
        neu = scores.get("neutral", 0)
        neg = scores.get("negative", 0)

        return f"""
        <div class="sentiment-bar-box">
            <div style="margin-bottom: 8px; font-weight: 600; text-align: center;">{aspect.capitalize() + " " + "Sentiment"}</div>
            <div style="display: flex; flex-direction: row; align-items: center; gap: 8px;"> <div style="height: 10px; border-radius: 5px; overflow: hidden; display: flex; flex-grow: 1; background-color: #e0e0e0; box-shadow: inset 0 1px 2px rgba(0,0,0,0.08);">
                    <div style="width: {pos}%; background-color: #222;"></div>
                    <div style="width: {neg}%; background-color: #666;"></div>
                    <div style="width: {neu}%; background-color: #999;"></div>
                </div>
                <div style="font-size: 12px; color: #333; white-space: nowrap;"> {pos}% / {neg}% / {neu}%
                </div>
            </div>
        </div>
        """


    def render_all_bars(self, sentiment_dict):
        all_bars_html = "\n".join(self.render_sentiment_bar(a, s) for a, s in sentiment_dict.items())
        return f"""
        <div class="sentiment-bars-container">
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

        json_output = json.dumps(self.retrieved_chunks_for_display, indent=2, ensure_ascii=False)
        return json_output, gr.update(visible=True)

    # Helper function to clear and hide chunks output
    def clear_and_hide_chunks(self):
        return gr.update(value="", visible=False)


    """ Create the Gradio interface
    This section sets up the Gradio interface with a clean and minimalist design.
    It includes a title, input textbox for queries, buttons for generating summaries and displaying chunks,
    and output textboxes for the summary and chunks.
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
            margin-bottom: 20px !important; /* Added margin below buttons */
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
            margin-bottom: 20px !important; /* Added margin below sentiment bars */
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

        .gr-section-spacing {
            margin-bottom: 25px !important; /* General spacing between major sections */
        }

        .gr-top-margin {
            margin-top: 25px !important; /* Margin for elements that need space above them */
        }

        .gradio-container > .block {
            padding: 10px !important; 
        }

        .gradio-container {
            padding: 20px !important;
        }

        .sentiment-legend-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            
            padding: 10px;
            border: 1px solid #eee;
            border-radius: 8px;
            gap: 15px; /* Space between legend items */
            margin-top: 15px !important;
        }

        .legend-item {
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #333;
        }

        .legend-color-box {
            width: 20px;
            height: 20px;
            border: 1px solid #ccc;
            border-radius: 4px;
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

            with gr.Row(elem_classes="gr-section-spacing"):
                with gr.Column(scale=4):
                    summary_output = gr.Textbox(show_label=False, lines=6, interactive=False)

            with gr.Row(): # This row holds sentiment display and the new legend
                with gr.Column(scale=1): # Column for sentiment bars
                    sentiment_display = gr.HTML(label="Sentiment Breakdown", elem_classes="sentiment-bars-container")
            with gr.Row():
                with gr.Column(scale=2): # Column for legend (adjust scale as needed)
                        sentiment_legend_html = gr.HTML(
                        """
                        <div class="sentiment-legend-container">
                            <div class="legend-item">
                                <div class="legend-color-box" style="background-color: #222;"></div>
                                Positive
                            </div>
                            <div class="legend-item">
                                <div class="legend-color-box" style="background-color: #666;"></div>
                                Negative
                            </div>
                            <div class="legend-item">
                                <div class="legend-color-box" style="background-color: #999;"></div>
                                Neutral
                            </div>
                            <div class="legend-item">
                                <div class="legend-color-box" style="background-color: #e0e0e0;"></div>
                                Not Defined
                            </div>
                        </div>
                        """,
                        label="Legend",
                        visible=False
                        )

            chunks_output = gr.Code(language="json", visible=False, interactive=False, elem_classes="gr-top-margin")

            generate_button.click(
                self.clear_and_hide_chunks,
                inputs=[],
                outputs=[chunks_output]
            ).then(
                self.generate_summary_stream,
                inputs=query_input,
                outputs=[summary_output, sentiment_display, sentiment_legend_html]
            )

            show_chunks_button.click(self.display_chunks, inputs=[], outputs=[chunks_output, chunks_output])

        demo.launch()

if __name__ == "__main__":
    user_interface = user_interface()
    user_interface.launch_interface()
