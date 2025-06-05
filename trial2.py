import csv
from sentence_transformers import SentenceTransformer

def load_and_embed_data(filename, limit_reviews=50, truncate_words=50):
    selected_data = []

    # Step 1: Load CSV and extract 'Model' and 'comment'
    with open(filename, encoding='utf-8') as reviewset:
        reviewset_data = csv.DictReader(reviewset)

        for row in reviewset_data:
            if len(selected_data) >= limit_reviews:
                break

            model_name = row.get("Model", "").strip()
            comment_text = row.get("comment", "").strip()

            # ✅ Only process rows with non-empty model and comment
            if not model_name or not comment_text:
                continue

            words = comment_text.split()
            truncated_comment = " ".join(words[:truncate_words])
            selected_data.append((model_name, truncated_comment))

    print(f"\n✅ Selected {len(selected_data)} valid rows.")

    Model = SentenceTransformer('all-MiniLM-L6-v2')
    results = []

    # Step 2: Treat each row as its own chunk
    for i, (model_name, comment) in enumerate(selected_data):
        print(f"\n📄 Processing row {i+1}:")
        print(f" - Model: {model_name}")
        print(f" - Comment: {comment[:60]}...")

        embedding = Model.encode([comment])[0]  # Single input
        results.append([(model_name, embedding)])  # Still keeping chunk format for consistency

       # ✅ Print the full embedding
        print(f"✅ Row {i+1} embedded. Full embedding:\n{embedding}")

    print(f"\n🎯 All rows processed as individual chunks. Total rows: {len(results)}")

    return results

if __name__ == "__main__":
    load_and_embed_data("reviewset.csv")  # Replace with your actual CSV filename
