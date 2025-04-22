import torch
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
import json
from ms_marco_dataset import MSMarcoWord2VecDataset

# No need for nltk or punkt anymore

# Load saved artifacts
w2v_model = Word2Vec.load("saved_artifacts/msmarco_word2vec_full.model")
with open("saved_artifacts/triples_full.json", encoding="utf-8") as f:
    triples = json.load(f)

# Create dataset
dataset = MSMarcoWord2VecDataset(triples, w2v_model, max_length=32)

# Create DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Fetch a single batch
batch = next(iter(loader))

# Print shapes
# Expected: [batch_size, max_len, embed_dim]
print("Query shape:", batch['query'].shape)
print("Pos Doc shape:", batch['pos_doc'].shape)
print("Neg Doc shape:", batch['neg_doc'].shape)

# Preview first query tensor
print("\nFirst Query Tensor:")
print(batch['query'][0])
