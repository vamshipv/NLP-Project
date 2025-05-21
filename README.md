#Project Overview

This project implements a Retriever-Generator architecture that allows users to query a document corpus and receive relevant, generated answers based on context. It combines the power of FAISS-based similarity search and a Transformer-based text generation model (FLAN-T5).

The system:
1.  Indexes text documents using vector embeddings (via SentenceTransformers).<br/>
2.  Retrieves the most relevant chunks of a document for a given query using FAISS.<br/>
3.  Generates human-readable answers to the user's question using the FLAN-T5 model.<br/>
4.  Logs responses with metadata like timestamp, similarity score, and document context.<br/>

#Requirements
Make sure you have Python installed and run the requirements file before running the main file<br/>
pip install -r requirements.txt

#Usage Instruction
1.  Navigate to the approriate directory to run the program, use python3 FileName.py to run the program <br/>
2.  Retriever Program is already loaded with default document on Cats <br/>
3.  User has the option to choose to load new document, overwrite and add to the existing document.<br/>
4.  Enter the User Query to find the results based on input <br/>

#Example Flow
Launch the program.<br/>

Choose to load cats.txt.<br/>

Ask a question like:<br/>
"What does Pooh do when he's sad?"<br/>

The system:<br/>

Retrieves relevant chunks.<br/>

Sends them to the generator.<br/>

Returns an answer like:<br/>
"Pooh often seeks honey or visits friends when he feels sad."<br/>

All logs are saved in generation_log.jsonl.<br/>

#How it works
1.  Retriever (retriever.py)<br/>
2.  Splits documents into chunks.<br/>
3.  Encodes with SentenceTransformer.<br/>
4.  Stores embeddings using FAISS index.<br/>
5.  Retrieves top-k similar chunks for a user query.<br/>

Generator (generator.py)<br/>
1.  Builds prompts with retrieved chunks.<br/>

2.  Uses google/flan-t5-base to generate answers.<br/>

3.  Logs session details with timestamp and relevance label.<br/>
