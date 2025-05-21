# Project Overview

This project implements a Retriever-Generator architecture that allows users to query a document corpus and receive relevant, generated answers based on context. It combines the power of FAISS-based similarity search and a Transformer-based text generation model (FLAN-T5).

The system:
1.  Indexes text documents using vector embeddings (via SentenceTransformers).<br/>
2.  Retrieves the most relevant chunks of a document for a given query using FAISS.<br/>
3.  Generates human-readable answers to the user's question using the FLAN-T5 model.<br/>
4.  Logs responses with metadata like timestamp, similarity score, and document context.<br/>

## Requirements
Make sure you have Python installed and run the requirements file before running the main file<br/>
**pip install -r requirements.txt**

## Usage Instruction
1.  Navigate to the approriate directory to run the program, use python3 FileName.py to run the program <br/>
2.  Retriever Program is already loaded with default document on Cats <br/>
3.  User has the option to choose to load new document, overwrite and add to the existing document.<br/>
4.  Enter the User Query to find the results based on input <br/>

## Example Flow
1.  Launch the program.<br/>
2.  Choose to load cats.txt.<br/>
3.  Ask a question like : "Who is pooh?"<br/>
4.  The system retrieves relevant chunks.<br/>
5.  Sends them to the generator.<br/>
6.  Returns an answer like : "a bear"<br/>
7.  All logs are saved in generation_log.jsonl.<br/>

## How it works
**Retriever (retriever.py)**<br/>
1.  Splits documents into chunks.<br/>
2.  Encodes with SentenceTransformer.<br/>
3.  Stores embeddings using FAISS index.<br/>
4.  Retrieves top-k similar chunks for a user query.<br/>

**Generator (generator.py)**<br/>
1.  Builds prompts with retrieved chunks.<br/>
2.  Uses google/flan-t5-base to generate answers.<br/>
3.  Logs session details with timestamp and relevance label.<br/>