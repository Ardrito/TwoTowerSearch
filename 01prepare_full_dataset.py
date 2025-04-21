import os
import json
import random
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from data.preprocessing import extract_query_doc_pairs, generate_triples
from data.tokenization import tokenize_texts

SAVE_DIR = "saved_artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {path}")

def train_word2vec(sentences, vector_size=300, window=5, min_count=2):
    print("🧠 Training Word2Vec model (CBOW)...")
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=0)
    return model

def build_doc_table(query_doc_pairs):
    doc_table = {}
    doc_to_id = {}
    doc_id_counter = 0

    for pair in query_doc_pairs:
        for passage in pair['all_passages']:
            if passage not in doc_to_id:
                doc_id = f"d{doc_id_counter:06d}"
                doc_to_id[passage] = doc_id
                doc_table[doc_id] = passage
                doc_id_counter += 1

    return doc_table, doc_to_id

if __name__ == "__main__":
    print("📦 Loading full MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")  # full training set

    print("🔍 Extracting query-document pairs...")
    all_pairs = extract_query_doc_pairs(dataset["train"])
    print(f"✅ Extracted {len(all_pairs)} pairs.")

    print("📚 Building document table...")
    doc_table, doc_to_id = build_doc_table(all_pairs)
    print(f"✅ Unique passages: {len(doc_table)}")

    print("🔁 Generating query-positive-negative triples...")
    all_triples = generate_triples(all_pairs, doc_to_id, num_negatives=1)

    print(f"✅ Generated {len(all_triples)} triples.")

    print("✂️ Tokenizing all triples...")
    tokenized_sentences = tokenize_texts(all_triples)
    print(f"✅ Tokenized {len(tokenized_sentences)} sentences.")

    print("🧠 Training Word2Vec model (CBOW)...")
    w2v_model = train_word2vec(tokenized_sentences)

    print("💾 Saving Word2Vec model and artifacts...")
    w2v_model.save(os.path.join(SAVE_DIR, "msmarco_word2vec_full.model"))
    w2v_model.wv.save_word2vec_format(os.path.join(SAVE_DIR, "msmarco_vectors_full.vec"), binary=False)

    save_json(tokenized_sentences, os.path.join(SAVE_DIR, "tokenized_sentences_full.json"))
    save_json(all_triples, os.path.join(SAVE_DIR, "triples_full.json"))
    save_json(doc_table, os.path.join(SAVE_DIR, "doc_table.json"))

    print("✅ Full dataset preparation complete!")
    print("💾 All artifacts saved in 'saved_artifacts' directory.")
