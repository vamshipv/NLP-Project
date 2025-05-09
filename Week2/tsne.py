from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the sentences from data.txt
with open('/Users/dechammacg/Documents/NLPPro/NLProc-Proj-M-SS25/baseline/data/data.txt', 'r') as file:
    sentences = [line.strip() for line in file if line.strip()]

# Create embeddings for sentences
embeddings = model.encode(sentences)

#Cosine similarity
similarity_matrix = cosine_similarity(embeddings)

# Print similarities
print("Cosine Similarity Matrix:\n")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = similarity_matrix[i][j]
        print(f"'{sentences[i]}' <-> '{sentences[j]}' = {sim:.4f}")
print()

#using t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)
plt.figure(figsize=(12, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color='blue')
for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (reduced_embeddings[i, 0] + 0.2, reduced_embeddings[i, 1] + 0.2))

plt.title("Visualization for Sentence Embeddings")
plt.grid(True)
plt.show()
