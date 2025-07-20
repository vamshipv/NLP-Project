# Product Review Summarizer

A pipeline for summarizing customer reviews on specific products using dense semantic search, product title matching, and generative summarization.

---

## Overview

This project implements a Retriever-Generator architecture designed to provide concise, review-based summaries. It leverages:

* **FAISS** for fast vector similarity search
* **SentenceTransformers** for embedding customer reviews
* **Gemma 2:2B** for generative text summarization
* **flashtext** for robust product title matching
* **Gradio** for an intuitive and interactive user interface

With this system, users can query real product feedback—for example, by asking for a summary on "Samsung Galaxy M01"—and receive a concise, review-based summary powered by **semantic search** and **Gemma 2:2B** text generation.

---

## Features

* **Title Matching**: Utilizes `flashtext` to accurately match product titles, ignoring variations in color, RAM, or storage.
* **Efficient Retrieval**: Employs a FAISS index for high-performance dense retrieval of relevant review snippets.
* **Sentiment Analysis**: Analyzes customer reviews to provide sentiment context, enhancing the quality of generated summaries.
* **Aspect-Based Summarization**: Allows users to focus on specific aspects of a product (e.g., camera, battery life) for more targeted summaries.
* **Natural Language Summarization**: Generates fluent summaries using the Gemma 2:2B model.
* **Interactive UI**: Provides a user-friendly Gradio interface for seamless querying and exploration.
* **Transparent Logging**: Logs every retrieval and summarization step, including the original query, generated summary, and source chunks for full context.

---

## Setup

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

### Additional Requirement: Ollama for Gemma

Before running the generator, you'll need to install [Ollama](https://ollama.com/) to handle Gemma-based summarization locally.

1.  **Download and Install Ollama**: Visit [https://ollama.com/download](https://ollama.com/download) and follow the installation instructions for your operating system.
2.  **Pull the Gemma Model**: Once Ollama is installed, open your terminal and pull the Gemma 2B model:

    ```bash
    ollama pull gemma2:2b
    ```

---

## How It Works

The system operates as a modular Retriever-Generator pipeline, processing customer reviews through several key stages:

---

### 1. Chunking & Embedding

Customer reviews (structured with `Brand`, `Model`, `Stars`, `Comment`) are loaded from a JSON file. These reviews are then:

* Split into approximately 512-token chunks using NLTK.
* Embedded into vector representations using the `intfloat/e5-base-v2` SentenceTransformer model.
* Stored in a FAISS index to enable efficient similarity-based search.

---

### 2. flashtext Product Matching

When a user submits a query, the system:

* Cleans and simplifies the query (e.g., removes phrases like "summary on" or "feedback for").
* Performs a match against a list of known product titles using flashtext.
* Canonicalizes matched titles by removing specific technical details (like RAM, color, or storage variants).
* Filters the review chunks to include only those related to any matching product variant.

---

### 3. Semantic Retrieval

With the product-specific chunks identified, the system then:

* Embeds the cleaned user query using the same `e5-base-v2` model.
* Constructs a temporary FAISS index containing only the matched product chunks.
* Executes a top-k similarity search on this temporary index to retrieve the most semantically relevant review snippets.

---

### 4. Summarization with Gemma

The retrieved review chunks are then used to:

* Construct a detailed prompt.
* Send this prompt to the `gemma2:2b` language model, which runs locally via Ollama.
* Generate and return a concise, fluent summary based on the provided review content.
* If the user specifies an aspect (e.g., "camera"), the summarization focuses on that aspect, using sentiment analysis to guide the summary generation.
* If no reviews are found for the specified product, the system returns a message indicating that no relevant reviews were available.
---

### 5. Sentiment Analysis
The system performs sentiment analysis on the retrieved review chunks to provide context for the summary. 
* It categorizes reviews into positive, neutral, and negative sentiments, which are then displayed alongside the summary.* The sentiment analysis is performed using the `cardiffnlp/twitter-roberta-base-sentiment` model, which provides a simple and effective way to analyze the sentiment of text.
* The sentiment scores are used to create a breakdown of positive, neutral, and negative sentiments for the product.
* This breakdown is displayed in the user interface, allowing users to see the sentiment distribution across the reviews.
* The sentiment analysis results are also logged for transparency and debugging purposes.

---

### 6. User Query Handling
The user query is processed to extract the product name and any specific aspects of interest. The system handles variations in product names, ensuring that the summarization is relevant to the user's request.

* The query is cleaned to remove unnecessary phrases (e.g., "summary on", "feedback for").
* It acts as layer between the user interface and the summarization logic, ensuring that the user input is correctly interpreted and processed.
* If a specific aspect is mentioned in the query, the summarization focuses on that aspect, using sentiment analysis to guide the summary generation.
* If the aspect is not specified, the system generates a general summary of the product based on all available reviews.
* If a product is not found in the reviews, the system returns a message indicating that no relevant reviews were available.
---

### 7. User Interface
The user interface is built using Gradio, providing an interactive web application where users can:
* Input their product summarization queries.
* View the generated summary.
* Review the specific review chunks used to create the summary.
* Explore sentiment breakdowns for different aspects of the product.
---

### 8. Logging

All operations are thoroughly logged for transparency and debugging:

* `summary_log.json`: This file records the original query, the generated summary, and all source chunks used to create the summary.

---

### 9. Test Cases
The system includes a set of test cases to validate the summarization functionality. These tests cover various product queries and aspects, ensuring that the summarization logic works as expected across different scenarios.

---

### 10. Evaluation
The system is evaluated based on the F1 score of the generated summaries against a set of reference summaries.

---

## Usage

To run the application, clone the repository and navigate to the `baseline/user_interface` directory and execute the `user_interface.py` script:

```bash
cd Project/baseline/user_interface
python user_interface.py
```

Upon successful execution, the application will launch a Gradio interface, typically accessible via a `localhost` URL (e.g., `http://127.0.0.1:7860`). Open this link in your web browser.

### Example Interaction

1.  **Input a Query**: In the Gradio interface, type your product summarization query, for instance:
    ```
    Summary on Samsung Galaxy M01
    ```
2.  **Get Summary**: The system will process your request and display a concise summary of the relevant customer reviews.
3.  **Review Chunks**: Below the summary, you will find "Chunks". Clicking on "Show Chunks" button will reveal the specific review chunks that were used by the Gemma model to generate the summary, allowing you to review the source content.
4. **Sentiment Breakdown**: The interface will also display a sentiment breakdown for the product, showing the distribution of positive, neutral, and negative sentiments across the reviews.
5. **More Test Examples**: You can find more test examples in the `test_queries.txt` file. These queries can be used to test the summarization functionality with various products and aspects.

