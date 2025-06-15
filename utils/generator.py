import torch
import json
import logging
from datetime import datetime
import sys 
import os  

# Updated imports to use T5Tokenizer and T5ForConditionalGeneration again for Flan-T5-Large
from transformers import T5Tokenizer, T5ForConditionalGeneration 
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging to append to a JSON file
logging.basicConfig(filename="summary_log.json", level=logging.INFO, format="%(message)s")

class Generator:
    def __init__(self):


        self.text_rank = TextRankSummarizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Using Flan-T5-Large (770M parameters) for better performance on limited VRAM/CPU
        self.model_name = "google/flan-t5-large" 
        
        try:
            print(f"Loading tokenizer for: {self.model_name}...")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            print(f"Loading model for: {self.model_name}...")
            
            # Load the model. Check for CUDA availability and move model to GPU if present.
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                print("CUDA is available. Moving model to GPU.")
                self.model.to("cuda")
            else:
                print("CUDA not available. Running model on CPU. Performance may still be slower than GPU, but significantly better than Flan-T5-XL on CPU.") 

        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
            raise

    def extractive_summary(self, text, sentence_count=3): 
        # This method is still here, but its output will no longer be the direct input to abstractive_summary
        if not text.strip():
            return ""

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        raw_summary_sentences = [str(sentence) for sentence in self.text_rank(parser.document, sentence_count * 2)]

        positive_sentences = []
        negative_sentences = []
        neutral_sentences = []

        for s in raw_summary_sentences:
            sentiment_score = self.sentiment_analyzer.polarity_scores(s)['compound']
            if sentiment_score > 0.2:
                positive_sentences.append(s)
            elif sentiment_score < -0.2:
                negative_sentences.append(s)
            else:
                neutral_sentences.append(s)

        final_summary_sentences = []
        added_sentences = set()

        def add_unique_sentences(source_list):
            for s in source_list:
                if len(final_summary_sentences) >= sentence_count:
                    break
                if s not in added_sentences:
                    final_summary_sentences.append(s)
                    added_sentences.add(s)

        add_unique_sentences(positive_sentences)
        add_unique_sentences(negative_sentences)
        add_unique_sentences(neutral_sentences)

        return " ".join(final_summary_sentences)

    def abstractive_summary(self, combined_text_input): # Changed method signature to accept single string
        if not combined_text_input.strip():
            return "No review content available for summarization."

        inputs = self.tokenizer(
            combined_text_input, # Pass combined string directly
            return_tensors="pt", truncation=True, max_length=1024
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs['input_ids'],  
                attention_mask=inputs['attention_mask'], 
                max_length=350, 
                min_length=30,  
                num_beams=6,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def generate_summary(self, user_query, chunk_list):
        if not chunk_list:
            return "No relevant reviews found."

        narrative_chunks = chunk_list[:15]
        narrative = " ".join([chunk['text'].strip() for chunk in narrative_chunks])

        # Extractive summary is kept for logging but not used as direct input for abstractive model
        extracted_summary = self.extractive_summary(narrative, sentence_count=3) 

        # Streamlined prompt as instructions, with narrative explicitly labeled within the single input string
        prompt_and_narrative_combined = (
            f"Synthesize customer opinions on the {user_query}. "
            "Extract and combine key information from the following reviews into a coherent summary. "
            "The summary should have the following sections:\n"
            "1. Overall Impression\n"
            "2. Key Positive Aspects\n"
            "3. Key Negative Aspects\n"
            "4. Conclusion on whether it's a good buy.\n"
            "Use a neutral, third-person perspective, avoiding direct quotes. "
            f"Customer Reviews: {narrative}" # CHANGED: Combining into one string with a clear label
        )
        
        # Pass the single combined string to abstractive_summary
        abstractive_summary_result = self.abstractive_summary(prompt_and_narrative_combined) 

        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model_name,
            "user_query": user_query,
            "num_chunks_processed": len(narrative_chunks),
            "original_chunks": [chunk['text'] for chunk in narrative_chunks],
            "extracted_summary": extracted_summary, # Log the extractive summary for reference
            "final_abstractive_summary": abstractive_summary_result,
        }

        with open("summary_log.json", "a", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
            f.write("\n")

        return abstractive_summary_result

def main():
    generator = Generator()

if __name__ == "__main__":
    main()
