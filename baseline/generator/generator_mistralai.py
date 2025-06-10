# generator/generator.py

from typing import List

class Generator:
    def __init__(self, model_type="huggingface", hf_model="mistralai/Mistral-7B-Instruct-v0.1"):
        self.model_type = model_type
        self.hf_model = hf_model

        if model_type == "huggingface":
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        elif model_type == "llama-cpp":
            from llama_cpp import Llama
            self.llm = Llama(model_path="path/to/your/local/llama-model.gguf")

        else:
            raise ValueError("Model type must be 'huggingface' or 'llama-cpp'")

    def build_prompt(self, chunks: List[str], question: str) -> str:
        """
        Builds a prompt from the retrieved text chunks and the user's question.
        """
        context = "\n\n".join(chunks)
        prompt = (
            f"You are a helpful assistant summarizing product reviews.\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return prompt

    def generate_answer(self, prompt: str) -> str:
        """
        Generates an answer using the initialized model.
        """
        if self.model_type == "huggingface":
            outputs = self.pipeline(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            return outputs[0]["generated_text"].split("Answer:")[-1].strip()

        elif self.model_type == "llama-cpp":
            output = self.llm(prompt, max_tokens=200)
            return output['choices'][0]['text'].strip()
