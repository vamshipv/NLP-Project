import os
import sys
import json
from bert_score import score as bert_score
from rouge_score import rouge_scorer

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# Add the generator path to import Generator class
sys.path.append(os.path.abspath(os.path.join('..', 'baseline', 'generator')))
from generator import Generator

def evaluate_summary(user_query, retrieved_chunks, reference_summary, aspect=None):
    """
    Generate a summary using Generator and evaluate it with ROUGE-L and BERTScore.

    Args:
        user_query (str): Query to generate the summary.
        retrieved_chunks (list of str): List of review chunks.
        reference_summary (str): Reference summary for evaluation.

    Returns:
        float: BERTScore F1 of the generated summary.
    """
    # review_list = [{"text": chunk} for chunk in retrieved_chunks]
    review_list = retrieved_chunks
    generator = Generator()
    generated_summary, _ = generator.generate_summary(user_query, review_list, aspect=None)

    print("\n--- Generated Summary ---\n")
    print(generated_summary)

    # Calculate ROUGE-L
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(reference_summary, generated_summary)
    rouge_l = rouge_scores["rougeL"]

    if aspect == None:
        print("General evaluation")
    else:
        print("Aspect based evaluation")

    print("\n--- ROUGE-L Score ---")
    print(f"Precision: {rouge_l.precision:.4f}")
    print(f"Recall:    {rouge_l.recall:.4f}")
    print(f"F1-Score:  {rouge_l.fmeasure:.4f}")

    # Calculate BERTScore
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
    user_query_1 = "summarize to me the user reviews about samsung galaxy m51 "
    retrieved_chunks_1 = [
        {
            "text": "After using for more than 2 day's I must say, snapdragon 730g works wonderful. CAMERA. This time M series did a honest job by giving us a INTELLI camera of 64 mega Pixcel and more feature in selfie camera. DISPLAY & FINGERPRINT. With Samsung SuperAmoled display you can truly trust the amazing view angle.",
            "brand": "Samsung",
            "model": "Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)",
            "stars": "5"
        },
        {
            "text": "Camera quality is really gud wth many features which luks tempting which im yet to explore. Even in dim light it scores quite well. Have all latest features like slow mo, 10x zoom, focus lens, panorma view and other added features. Battery ofcourse was the highlight while buying ths phn.",
            "brand": "Samsung",
            "model": "Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)",
            "stars": "5"
        },
        {
            "text": "Awesome camera Awesome battery Awesome display .... Super clear Zoom Super Macro sensor Super ultrawide angle Atlast Thanks Samsung for 730G",
            "brand": "Samsung",
            "model": "Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)",
            "stars": "5"
        },
        {
            "text": "This is a perfect mobile for you, if your first preference is battery, and decent level of camera quality, and if you need all trending features to be covered in non Chinese mobile, THEN GO FOR IT...",
            "brand": "Samsung",
            "model": "Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)",
            "stars": "5"
        },
        {
            "text": "I found it reliable for my 1920p to 4k level of movie watching without any distraction due to Infinite O display. Fingerprint is fast. Classy look, strong built for this budget amount. 6.7 inches display is good to handle.",
            "brand": "Samsung",
            "model": "Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)",
            "stars": "5"
        },
        {
            "text": "Ok so here comes the first review of this phone. Ordered on the first day of its sale. Seamless fingerprint scanner. Side fingerprint scanner really makes it stylish and too easy to handle as well.",
            "brand": "Samsung",
            "model": "Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)",
            "stars": "5"
        },
        {
            "text": "Camera not much good that I am expecting. I am not happy overall performance of this phone at this price band... Display quality is just good not best. Sound quality is good.",
            "brand": "Samsung",
            "model": "Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)",
            "stars": "1"
        },
        {
            "text": "Cons:_Overpriced. Battery drain only fast in normal uses gives only 24 hours backup. Build quality is not good.",
            "brand": "Samsung",
            "model": "Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)",
            "stars": "1"
        },
        {
            "text": "Battery: Battery is 7K mha but it is draining so fast. Camera is just fine",
            "brand": "Samsung",
            "model": "Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)",
            "stars": "3"
        },
        {
            "text": "When you focus on Sky whole pic turn into blue. Saturation level is too much high in this phone. Brightness fluctuates fast too much automatically.",
            "brand": "Samsung",
            "model": "Samsung Galaxy M51 (Electric Blue, 6GB RAM, 128GB Storage)",
            "stars": "2"
        }
    ]

    
    reference_summary_1 = (
        "Customer reviews of the Samsung Galaxy M51 exhibit overwhelmingly positive sentiment, particularly praising its impressive battery life, camera quality, and user interface.  While some reviewers note minor drawbacks in build quality, performance, and sound quality, these criticisms are overshadowed by overall satisfaction with aspects like display clarity and feature set.  A consistent theme appears in reviews highlighting the value offered by the device's features at a competitive price point." 
         
        "\n\n\nOVERALL SENTIMENT : Positive"

        "\n- 67% of customers have positive reviews about battery, camera, display, performance."

        "\n- 25% have negative reviews about battery, camera, performance."

        "\n- 8% have neutral reviews about battery, performance. "
    )

    print("\n--- Reference Summary ---\n", reference_summary_1)
    evaluate_summary(user_query_1, retrieved_chunks_1, reference_summary_1, aspect=None)


    user_query_2 = "how is the battery of mi poco m2 pro"

    retrieved_chunks_2 = [
        {
            "text": "Very Good Phone under this budget. 5k +mAh battery. Delivery delayed because of Covid.",
            "brand": "MI",
            "model": "MI Poco M2 Pro (Green and Greener, 6GB RAM, 64GB Storage)",
            "stars": "3",
            "aspect": "battery"
        },
        {
            "text": "Good build quality. Sharp display. Powerful processor. Solid battery life. Decent daylight camera performance. Fast charging. No heating issue.",
            "brand": "MI",
            "model": "MI Poco M2 Pro (Green and Greener, 6GB RAM, 64GB Storage)",
            "stars": "5",
            "aspect": "battery"
        },
        {
            "text": "Super performance. But it was power off while using without any reason.",
            "brand": "MI",
            "model": "MI Poco M2 Pro (Green and Greener, 6GB RAM, 64GB Storage)",
            "stars": "3",
            "aspect": "battery"
        },
        {
            "text": "The phone charges really fast.",
            "brand": "MI",
            "model": "MI Poco M2 Pro (Green and Greener, 6GB RAM, 64GB Storage)",
            "stars": "4",
            "aspect": "battery"
        },
        {
            "text": "Battery backup is excellent, lasts more than a day on moderate usage. Really happy with it!",
            "brand": "MI",
            "model": "MI Poco M2 Pro (Green and Greener, 6GB RAM, 64GB Storage)",
            "stars": "5",
            "aspect": "battery"
        },
        {
            "text": "Fast charging is amazing. It takes just about an hour to go from 0 to 100%. Perfect for busy schedules.",
            "brand": "MI",
            "model": "MI Poco M2 Pro (Green and Greener, 6GB RAM, 64GB Storage)",
            "stars": "5",
            "aspect": "battery"
        },
        {
            "text": "Battery is long-lasting and doesn’t drain fast even when streaming or gaming for hours.",
            "brand": "MI",
            "model": "MI Poco M2 Pro (Green and Greener, 6GB RAM, 64GB Storage)",
            "stars": "5",
            "aspect": "battery"
        },
        {
            "text": "Excellent battery performance! Supports my whole day without any worry of charging. Very reliable.",
            "brand": "MI",
            "model": "MI Poco M2 Pro (Green and Greener, 6GB RAM, 64GB Storage)",
            "stars": "5",
            "aspect": "battery"
        },
        {
            "text": "The battery gives more than a day's use with ease. Gaming, browsing, watching videos—no problem at all.",
            "brand": "MI",
            "model": "MI Poco M2 Pro (Green and Greener, 6GB RAM, 64GB Storage)",
            "stars": "4",
            "aspect": "battery"
        },
        {
            "text": "Battery optimization is on point. It feels like it lasts forever. Easily one of the best phones for battery in this range.",
            "brand": "MI",
            "model": "MI Poco M2 Pro (Green and Greener, 6GB RAM, 64GB Storage)",
            "stars": "5",
            "aspect": "battery"
        }
    ]

  
    reference_summary_2 = (
        "Customers generally express positive sentiment regarding the battery performance of the MI Poco M2 Pro.  Reviewers highlight its long-lasting battery life that can easily power through a full day of moderate usage without needing frequent recharges. The phone's fast charging capabilities are also widely appreciated, allowing users to quickly replenish the battery even when pressed for time.  The consensus is that the phone offers robust battery performance and reliability, earning positive feedback from a substantial number of customers. "
        "\n\n\nOverall Sentiment : Positive"
        "\n- 100% of customers have positive reviews."
        "\n- 0% of customers have negative reviews."
        "\n- 0% of customers have neutral reviews."
    )

    print("\n--- Reference Summary ---\n", reference_summary_2)
    evaluate_summary(user_query_2, retrieved_chunks_2, reference_summary_2, aspect="Battery")