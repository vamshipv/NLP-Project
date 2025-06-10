# generator/generator.py

from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class Generator:
    def __init__(self, model_type="huggingface", hf_model=None):
        self.model_type = model_type
        self.hf_model = hf_model

        if model_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                # as we are not using GPU this is not needed
                # device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        elif model_type == "llama-cpp":
            from llama_cpp import Llama
            self.llm = Llama(model_path="path/to/your/local/llama-model.gguf")
        else:
            raise ValueError("Model type must be 'huggingface' or 'llama-cpp'")
        
    def generate(self, user_input):
        prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_input}<|assistant|>\n"
        response = self.generator(prompt, return_full_text=False)
        return response[0]['generated_text']

    def build_prompt(self, chunks: List[str], question: str) -> str:
        context = "\n\n".join(chunks)
        return (
            "<|system|>\n"
            "You are a helpful assistant that summarizes product reviews and answers user questions.\n"
            "<|user|>\n"
            f"Context:\n{context}\n\nQuestion: {question}\n"
            "<|assistant|>\n"
        )


    def generate_answer(self, prompt: str) -> str:
        """
        Generates an answer using the initialized model.
        """
        if self.model_type == "huggingface":
            outputs = self.pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
            return outputs[0]["generated_text"].split("Answer:")[-1].strip()

        elif self.model_type == "llama-cpp":
            output = self.llm(prompt, max_tokens=200)
            return output['choices'][0]['text'].strip()
