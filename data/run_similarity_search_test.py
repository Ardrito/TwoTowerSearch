import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load test embeddings and raw text
query_embeddings = np.load("saved_artifacts/query_embeddings_test.npy")
doc_embeddings = np.load("saved_artifacts/doc_embeddings_test.npy")

with open("saved_artifacts/raw_queries_test.json", encoding="utf-8") as f:
    raw_queries = json.load(f)

with open("saved_artifacts/raw_docs_test.json", encoding="utf-8") as f:
    raw_docs = json.load(f)

# Search with a query index
query_index = 3  # Change this to try different queries
query_vector = query_embeddings[query_index].reshape(1, -1)
scores = cosine_similarity(query_vector, doc_embeddings)[0]

top_k = 10
top_indices = scores.argsort()[::-1][:top_k]

print(f"\n🔍 Query: {raw_queries[query_index]}\n")
print("📄 Top similar test documents:\n")
for rank, idx in enumerate(top_indices, 1):
    print(f"Top {rank} (Score: {scores[idx]:.4f}):\n{raw_docs[idx][:300]}...\n")
