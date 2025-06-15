import torch
import json
import logging
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration 
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(filename="summary_log.json", level=logging.INFO, format="%(message)s")

class Generator:
    def __init__(self):
        self.text_rank = TextRankSummarizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.model_name = "google/flan-t5-large" 
        
        try:
            print(f"Loading tokenizer for: {self.model_name}...")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            print(f"Loading model for: {self.model_name}...")
            
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                print("CUDA is available. Moving model to GPU.")
                self.model.to("cuda")
            else:
                print("CUDA not available. Running model on CPU.") 
        except Exception as e:
            print(f"Error during model loading: {e}")
            raise

    def preprocess_reviews(self, chunk_list):
        """ Groups reviews by sentiment before passing them to FLAN-T5. """
        if not chunk_list:
            return {"positive": [], "negative": [], "neutral": []}

        positive_reviews, negative_reviews, neutral_reviews = [], [], []

        for chunk in chunk_list:
            review_text = chunk["text"].strip()
            sentiment_score = self.sentiment_analyzer.polarity_scores(review_text)["compound"]

            if sentiment_score > 0.2:
                positive_reviews.append(review_text)
            elif sentiment_score < -0.2:
                negative_reviews.append(review_text)
            else:
                neutral_reviews.append(review_text)

        return {
            "positive": positive_reviews[:5],
            "negative": negative_reviews[:5],
            "neutral": neutral_reviews[:5]
        }

    def create_summary_prompt(self, user_query, review_groups):
        """
            #TODO
            Here adding more parameters for the model to undertand better
        """
        """ Creates a structured prompt for FLAN-T5 summarization. """
        return (
            f"Synthesize customer opinions on the {user_query}. Extract and combine key information into a coherent summary.\n\n"
            "**Overall Customer Sentiment:** Most buyers appreciate the convenience of having two USB ports for charging multiple devices. "
            "Some users mention that the compact design makes storage easy, and the additional light improves visibility. However, durability varies, "
            "with reports of the device not seating properly in the power outlet or failing to provide adequate charging output.\n\n"
            "**Key Positive Aspects:**\n"
            "- Many users find the **dual-port design useful**, reducing clutter while charging multiple devices.\n"
            "- The **compact size** makes it easy to store.\n"
            "- Several customers praise the **light indicator**, which helps locate the charger in dark environments.\n\n"
            "**Key Negative Aspects:**\n"
            "- Some buyers report that the charger **does not fit securely** in power outlets.\n"
            "- There are complaints about **insufficient charging output**, especially for Android devices.\n"
            "- Longevity concerns arise due to **inconsistent performance over time**.\n\n"
            "**Final Verdict:** While the product offers good functionality, buyers seeking **high-output charging for Android devices** may need an alternative."
        )

    def abstractive_summary(self, combined_text_input):
        """ Generates an abstractive summary using FLAN-T5. """
        if not combined_text_input.strip():
            return "No review content available for summarization."

        inputs = self.tokenizer(
            combined_text_input, 
            return_tensors="pt", truncation=True, max_length=1024
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            """
            #TODO
            Here the parameters can be changed to adjust the length of the summary
            """
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],  
                attention_mask=inputs["attention_mask"], 
                max_length=600, 
                min_length=200,  
                num_beams=9,
                early_stopping=True,
                no_repeat_ngram_size=2  # Prevents excessive compression
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def generate_summary(self, user_query, chunk_list):
        """ Main function to generate a structured customer review summary. """
        if not chunk_list:
            return "No relevant reviews found."

        review_groups = self.preprocess_reviews(chunk_list)
        structured_prompt = self.create_summary_prompt(user_query, review_groups)
        final_summary = self.abstractive_summary(structured_prompt)

        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model_name,
            "user_query": user_query,
            "num_chunks_processed": len(chunk_list),
            "original_chunks": [chunk["text"] for chunk in chunk_list],
            "structured_prompt": structured_prompt,
            "final_abstractive_summary": final_summary
        }

        with open("summary_log.json", "a", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
            f.write("\n")

        return final_summary

def main():
    generator = Generator()

if __name__ == "__main__":
    main()
