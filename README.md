# Product Review Summarizer

A pipeline for summarizing customer reviews on specific products using dense semantic search, fuzzy product title matching, and generative summarization.

---

## Overview

This project implements a Retriever-Generator architecture designed to provide concise, review-based summaries. It leverages:

* **FAISS** for fast vector similarity search
* **SentenceTransformers** for embedding customer reviews
* **Gemma 2:2B** for generative text summarization
* **RapidFuzz** for robust fuzzy product title matching
* **Gradio** for an intuitive and interactive user interface

With this system, users can query real product feedback—for example, by asking for a summary on "Samsung Galaxy M01"—and receive a concise, review-based summary powered by **semantic search** and **Gemma 2:2B** text generation.

---

## Features

* **Intelligent Title Matching**: Utilizes `RapidFuzz` to accurately match product titles, ignoring variations in color, RAM, or storage.
* **Efficient Retrieval**: Employs a FAISS index for high-performance dense retrieval of relevant review snippets.
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

### 2. Fuzzy Product Matching

When a user submits a query, the system:

* Cleans and simplifies the query (e.g., removes phrases like "summary on" or "feedback for").
* Performs a fuzzy match against a list of known product titles using RapidFuzz.
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

---

### 5. Logging

All operations are thoroughly logged for transparency and debugging:

* `summary_log.json`: This file records the original query, the generated summary, and all source chunks used to create the summary.
