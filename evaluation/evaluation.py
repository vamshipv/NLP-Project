import os
import sys
import json
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# Ensure the project root is in the path
# sys.path.append(os.path.abspath(os.path.join("..", "generator")))
sys.path.append(os.path.abspath(os.path.join('..', 'baseline', 'generator')))
from generator import Generator

"""
This evaluation module is designed to:
1. Evaluate the generated summary against a reference summary using ROUGE-L and BERTScore.
2. Generate a summary based on user queries and retrieved chunks of reviews.
It uses the `Generator` class to create summaries and the `rouge_scorer` and `bert_score` libraries for evaluation."""
def evaluate_summary(user_query, retrieved_chunks, reference_summary):
    """
    Generate a summary and evaluate it using both ROUGE-L and BERTScore.
    """
    review_list = [{"text": chunk} for chunk in retrieved_chunks]
    generator = Generator()
    generated_summary = generator.generate_summary(user_query, review_list)
    print("\n--- Generated Summary ---\n")
    print(generated_summary)

    # ROUGE-L Score
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(reference_summary, generated_summary)
    rouge_l = rouge_scores["rougeL"]

    print("\n--- ROUGE-L Score ---")
    print(f"Precision: {rouge_l.precision:.4f}")
    print(f"Recall:    {rouge_l.recall:.4f}")
    print(f"F1-Score:  {rouge_l.fmeasure:.4f}")

    # BERTScore (uses default model - multilingual or 'roberta-large' if available)
    P, R, F1 = bert_score([generated_summary], [reference_summary], lang='en', verbose=False)
    print("\n--- BERTScore ---")
    print(f"Precision: {P[0]:.4f}")
    print(f"Recall:    {R[0]:.4f}")
    print(f"F1-Score:  {F1[0]:.4f}")

    return F1[0].item()


"""
This script evaluates the generated summary against a reference summary using ROUGE-L and BERTScore.
It initializes the `Generator` class, retrieves relevant chunks based on a user query, and generates a summary.
It then prints the generated summary and the evaluation scores.
"""
if __name__ == "__main__":
    user_query = "Summarize customer opinions on the Vivo Y91, focusing on battery life, camera performance, and build quality."

    retrieved_chunks = [
        "The phone has a long-lasting battery life.",
        "Camera is underwhelming in low light.",
        "Disappointed with the build quality.",
        "Front camera is worst, not up to the mark. Waste of money.",
        "Amazing screen length and audio. Camera works good for me. Brilliant auto sensors. Within the price range."
    ]

    reference_summary = (
        "The Vivo Y91 features impressive all-day battery life that satisfies most users. "
        "The camera is generally well-regarded, especially for its performance in low light. "
        "Some users find the build quality disappointing considering the price. "
        "The front camera falls short of expectations in quality. "
        "The phone’s screen and audio are considered strong points."
    )

    print("\n--- Reference Summary ---\n", reference_summary)
    evaluate_summary(user_query, retrieved_chunks, reference_summary)
