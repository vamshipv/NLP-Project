# baseline/main/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
index_path = os.path.join(base_dir,"retriever", "reviews_faiss.index")
text_path = os.path.join(base_dir,"retriever", "reviews_combined_texts.json")

# Load retriever
retriever = Retriever()
retriever.load(index_path, text_path)

# Ask a question
question = input("Ask your question: ")

# Get top chunks from retriever
top_chunks = retriever.get_top_chunks(question, top_k=3)

# Initialize Generator with distilBART
generator = Generator(hf_model="sshleifer/distilbart-cnn-12-6")

# Build the prompt (text context)
prompt = generator.build_prompt(top_chunks, question)

# Generate the answer
answer = generator.generate_answer(prompt)

# Print the result
print("\n📌 Answer:")
print(answer)
