from torch.utils.data import DataLoader
from gensim.models import Word2Vec
import json
import numpy as np
from ms_marco_dataset import MSMarcoWord2VecDataset

# Load test triples
with open("saved_artifacts/triples_test.json", encoding="utf-8") as f:
    triples = json.load(f)

# Load trained Word2Vec model
w2v_model = Word2Vec.load("saved_artifacts/word2vec_skipgram_text8.model")

# Prepare dataset
dataset = MSMarcoWord2VecDataset(triples, w2v_model, max_length=32)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

query_embeddings = []
doc_embeddings = []
raw_queries = []
raw_docs = []

print("🔄 Generating test average embeddings...")
for batch in loader:
    q = batch['query'].squeeze(0).mean(dim=0).numpy()
    d = batch['pos_doc'].squeeze(0).mean(dim=0).numpy()
    query_embeddings.append(q)
    doc_embeddings.append(d)

    raw_queries.append(triples[len(query_embeddings) - 1]['query'])
    raw_docs.append(triples[len(doc_embeddings) - 1]['pos_doc'])

# Save embeddings
np.save("saved_artifacts/query_embeddings_test.npy", query_embeddings)
np.save("saved_artifacts/doc_embeddings_test.npy", doc_embeddings)

with open("saved_artifacts/raw_queries_test.json", "w", encoding="utf-8") as f:
    json.dump(raw_queries, f, indent=2, ensure_ascii=False)

with open("saved_artifacts/raw_docs_test.json", "w", encoding="utf-8") as f:
    json.dump(raw_docs, f, indent=2, ensure_ascii=False)

print("✅ Test embeddings saved!")
