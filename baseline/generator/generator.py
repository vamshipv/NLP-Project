from zoneinfo import ZoneInfo
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import json
from datetime import datetime, UTC


class Generator:
    def __init__(self, model_name="google/flan-t5-base", max_tokens=100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def build_prompt(self, context: str, question: str) -> str:
        return f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion:\n{question}"

    def generate_answer(self, results: str,context: str, question: str, group_id: str ) -> str:
        nullInput = "Please enter something"
        if(question == ""):
            return nullInput
        prompt = self.build_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # self.log_run(question, results, context, prompt, answer, group_id)
        return answer
    
    def log_run(self, question, results, context, prompt, answer, group_id):
        log_file = "generation_log.jsonl"
        log_data = {
            "question": question,
            "retrieved_chunks": results,
            "prompt": prompt,
            "generated_answer": answer,
            "timestamp": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
            "group_id": group_id
        }

        # Check if file exists
        mode = "a" if os.path.exists(log_file) else "w"

        # Write (append or create)
        with open(log_file, mode) as f:
            f.write(json.dumps(log_data) + "\n")