from sentence_transformers import SentenceTransformer
import faiss
import numpy as np



#Load sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read the sentences from data.txt
with open('/Users/dechammacg/Documents/NLPPro/NLProc-Proj-M-SS25/baseline/data/data.txt', 'r') as file:
    sentences = [line.strip() for line in file if line.strip()]

#Create embeddings for the sentences
doc_embeddings = model.encode(sentences)

#Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings).astype('float32'))

#Define different formats of the same question
queries = [
    "Are books preferred over movies?",
    "Is reading a book chosen over watching a movie?"
]

#Perform similarity search for each query
top_k = 2

for q in queries:
    print(f"\n Query: {q}")
    q_embedding = model.encode([q])
    distances, indices = index.search(np.array(q_embedding).astype('float32'), top_k)

    for rank, idx in enumerate(indices[0]):
        print(f"  {rank+1}. {sentences[idx]} (Distance: {distances[0][rank]:.4f})")
