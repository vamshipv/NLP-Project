import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from baseline.retriever.retriever_ollama import Retriever
from baseline.retriever.gen_ollama_gemma import summarize_with_gemma
from rouge_score import rouge_scorer


def evaluate_summary(retrieved_chunks, reference_summary):
    """
    Generate a summary from retrieved text chunks and evaluate it against a reference summary using ROUGE-L.

    Args:
        retrieved_chunks (list of str): List of text chunks (e.g., user reviews) to be summarized.
        reference_summary (str): The reference summary to be compared with the generated summary.

    Returns:
        float: The ROUGE-L F1 score indicating the similarity between the generated and reference summaries.

    """
    context = " ".join(retrieved_chunks)
    generated_summary = summarize_with_gemma(context, device_name)
    print("\nGenerated Summary:\n")
    print(generated_summary)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)

    rouge_l = scores['rougeL']
    print("\nROUGE-L Score:")
    print(f"Precision: {rouge_l.precision:.4f}")
    print(f"Recall:    {rouge_l.recall:.4f}")
    print(f"F1-Score:  {rouge_l.fmeasure:.4f}")

    return rouge_l.fmeasure


if __name__ == "__main__":
    retrieved_chunks = [
            "The phone has a long-lasting battery life.",
            "Camera is underwhelming in low light.",
            "Disappointed with the build quality.",
            "Front camera is worst, not up to the mark. Waste of money.",
            "Amazing screen length and audio. Camera works good for me. Brilliant auto sensors. Within the price range."
        ]
    device_name = "Vivo y91"   
    reference_summary = " The Vivo Y91 features impressive all-day battery life that satisfies most users. The camera is generally well-regarded, especially for its performance in low light. Some users find the build quality disappointing considering the price. The front camera falls short of expectations in quality. The phone’s screen and audio are considered strong points."
    print("\nReference Summary:\n",reference_summary)
    evaluate_summary(retrieved_chunks, reference_summary)