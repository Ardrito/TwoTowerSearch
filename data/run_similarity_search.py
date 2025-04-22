import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# === Load TF-IDF weighted embeddings ===
query_vectors = np.load("saved_artifacts/query_embeddings_skipgram.npy")
doc_vectors = np.load("saved_artifacts/doc_embeddings_skipgram.npy")

with open("saved_artifacts/raw_queries.json", encoding="utf-8") as f:
    queries = json.load(f)

with open("saved_artifacts/raw_docs.json", encoding="utf-8") as f:
    docs = json.load(f)

# === Pick a sample query ===
query_index = 5  # change this to try other queries
query_vec = query_vectors[query_index].reshape(1, -1)
similarities = cosine_similarity(query_vec, doc_vectors)[0]

# === Get Top-K similar docs ===
top_k = 5
top_indices = similarities.argsort()[::-1][:top_k]

print(f"\n🔍 Query: {queries[query_index]}")
print("\n📄 Top similar documents:")

for rank, idx in enumerate(top_indices):
    score = similarities[idx]
    doc = docs[idx]
    print(f"\nTop {rank+1} (Score: {score:.4f}):\n{doc[:300]}...")
