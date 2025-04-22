import torch
from torch.utils.data import Dataset

import pickle

# Load the TF-IDF vectorizer
with open("saved_artifacts/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Get token-to-IDF map
idf_scores = dict(
    zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))


class MSMarcoWord2VecDataset(Dataset):
    def __init__(self, triples, w2v_model, max_length=32):
        """
        Args:
            triples: A list of dictionaries with keys 'query', 'pos_doc', and 'neg_doc'
            w2v_model: A trained gensim Word2Vec model
            max_length: Number of tokens to use (pad or truncate)
        """
        self.triples = triples
        self.w2v = w2v_model
        self.dim = w2v_model.vector_size
        self.max_length = max_length

    def embed(self, text):
        tokens = text.lower().split()
        vectors = []
        weights = []

        for token in tokens:
            if token in self.w2v.wv:
                vec = torch.tensor(self.w2v.wv[token])
                tfidf = idf_scores.get(token, 1.0)  # fallback to 1.0
                vectors.append(vec * tfidf)
                weights.append(tfidf)

        if not vectors:
            vectors = [torch.zeros(self.dim)]
            weights = [1.0]

        tensor = torch.stack(vectors[:self.max_length])
        weights_tensor = torch.tensor(weights[:self.max_length]).unsqueeze(1)

        weighted_tensor = tensor * weights_tensor

        if tensor.size(0) < self.max_length:
            pad_len = self.max_length - tensor.size(0)
            padding = torch.zeros(pad_len, self.dim)
            weighted_tensor = torch.cat([weighted_tensor, padding])

        return weighted_tensor

    def __getitem__(self, idx):
        triple = self.triples[idx]
        return {
            'query': self.embed(triple['query']).float(),
            'pos_doc': self.embed(triple['pos_doc']).float(),
            'neg_doc': self.embed(triple['neg_doc']).float()
        }

    def __len__(self):
        return len(self.triples)
