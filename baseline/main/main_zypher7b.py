import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator

# debugger
# import pdb

# Set paths based on current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '..'))

index_path = os.path.join(base_dir, "retriever","reviews_faiss.index")
text_path = os.path.join(base_dir, "retriever","reviews_combined_texts.json")

# pdb.set_trace()

retriever = Retriever()
retriever.load(index_path, text_path)

question = input("Ask your question: ")
top_chunks = retriever.get_top_chunks(question, top_k=3)

gen = Generator(model_type="huggingface", hf_model="HuggingFaceH4/zephyr-7b-beta")
prompt = gen.build_prompt(top_chunks, question)
answer = gen.generate_answer(prompt)

print("Answer:", answer)
