from zoneinfo import ZoneInfo
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import json
from datetime import datetime, UTC
from sentence_transformers import SentenceTransformer


class Generator:
    """
    Generator class with methods to build prompt, generate answer and logging
    """
    def __init__(self, model_name="google/flan-t5-base", max_tokens=100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def build_prompt(self, context: str, question: str) -> str:
        """
        This function takes context and question and builds a prompt
        
        Args:
            context (str) :  textual context
            question (str) : user query
            
        Returns:
            (str) : returns the prompt
        """
        return f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion:\n{question}"

    def generate_answer(self, results: str,context: str, question: str, group_id: str ) -> str:
        """
        Generates an answer to a given question using the FLAN-T5 model based on the provided context
        
        Args :
            results (str) : retrieved text chunks
            context (str) : textual context
            question (str) : user query
            group_id (str) : team identifier
    
        Returns :
            (str) : the generated answer from the model
        """
        nullInput = "Please enter something"
        if(question == ""):
            return nullInput
        prompt = self.build_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        retriever_similarity = self.compute_retriever_similarity_score(context, question)
        self.log_run(question, results, context, prompt, answer, group_id, retriever_similarity)
        return answer
    
    def compute_retriever_similarity_score(self, context: str, question: str) -> float:
        """
        Computes a cosine similarity score between the context and the question embeddings
        
        Args :
            context (str) : textual context
            question (str) : user query
        
        Return :
            (float) : cosine similarity rounded to one decimal place
            
        """
        try:
            # context_embedding = self.embedder.encode([context])[0]
            # query_embedding = self.embedder.encode([question])[0]
            # return round(float(torch.cosine_similarity([context_embedding], [query_embedding])[0][0]), 1)
            context_embedding = torch.tensor(self.embedder.encode(context))
            question_embedding = torch.tensor(self.embedder.encode(question))

            # Add batch dimension so shapes are (1, embedding_dim)
            context_embedding = context_embedding.unsqueeze(0)
            question_embedding = question_embedding.unsqueeze(0)

            similarity = torch.cosine_similarity(context_embedding, question_embedding).item()
            return round(similarity, 1)
        except Exception as e:
            print({e},"Try again:")
    
    def log_run(self, question, results, context, prompt, answer, group_id, retriever_similarity):
        """
        Logs the complete details of each session to a JSONL file 
        
        Args :
            question (str) : user query
            results (str) : retrieved text chunks
            context (str) : textual context
            prompt (str) : final prompt given to the model
            answer (str) : generated answer from the model
            group_id (str) : team identifier
            retriever_similarity (float) : similarity score between the context and the question embeddings
        
        """
        log_file = "generation_log.jsonl"
        if retriever_similarity >= 0.9:
            label = "Very strong match — answer is clearly on-topic"
        elif retriever_similarity >= 0.7:
            label = "Good relevance"
        elif retriever_similarity >= 0.4:
            label = "Partial or vague relevance"
        else:
            label = "Likely irrelevant or hallucinated answer"
        log_data = {
            "question": question,
            "retrieved_chunks": results,
            "prompt": prompt,
            "generated_answer": answer,
            "timestamp": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
            "group_id": group_id,
            "retriever_similarity_score": retriever_similarity,
            "retriever_similarity_label" : label
        }

        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)  

        log_file = os.path.join(log_dir, log_file) 

        mode = "a" if os.path.exists(log_file) else "w"

        with open(log_file, mode) as f:
            f.write(json.dumps(log_data) + "\n")