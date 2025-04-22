import os
import json
from datasets import load_dataset
from gensim.models import Word2Vec
import gensim.downloader as api

from preprocessing import extract_query_doc_pairs, generate_triples
from tokenization import tokenize_texts

SAVE_DIR = "saved_artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {path}")


def train_word2vec(sentences, vector_size=300, window=5, min_count=2):
    print("🧠 Training Word2Vec model (Skip-Gram)...")
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # ✅ Skip-Gram
        workers=os.cpu_count()
    )
    return model


if __name__ == "__main__":
    print("📦 Loading full MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")

    print("🔍 Extracting query-document pairs...")
    all_pairs = extract_query_doc_pairs(dataset["train"])
    print(f"✅ Extracted {len(all_pairs)} pairs.")

    print("🔁 Generating query-positive-negative triples...")
    all_triples = generate_triples(all_pairs, num_negatives=1)
    print(f"✅ Generated {len(all_triples)} triples.")

    print("✂️ Tokenizing all triples...")
    tokenized_sentences = tokenize_texts(all_triples)
    print(f"✅ Tokenized {len(tokenized_sentences)} sentences.")

    # ✅ Returns tokenized sentences directly
    print("📥 Downloading & loading Text8 corpus...")
    text8_corpus = list(api.load("text8"))
    print(f"✅ Loaded {len(text8_corpus)} Text8 sentences.")

    print("➕ Combining MS MARCO with Text8 sentences...")
    combined_sentences = tokenized_sentences + text8_corpus
    print(f"✅ Combined corpus size: {len(combined_sentences)} sentences.")

    print("🧠 Training Word2Vec model (Skip-Gram on combined corpus)...")
    w2v_model = train_word2vec(combined_sentences)

    # Save everything
    print("💾 Saving Word2Vec model and artifacts...")
    w2v_model.save(os.path.join(SAVE_DIR, "word2vec_skipgram_text8.model"))
    w2v_model.wv.save_word2vec_format(
        os.path.join(SAVE_DIR, "word_vectors_skipgram_text8.vec"),
        binary=False
    )
    save_json(all_triples, os.path.join(SAVE_DIR, "triples_full.json"))
    save_json(tokenized_sentences, os.path.join(
        SAVE_DIR, "tokenized_sentences.json"))

    print("\n✅ Full dataset preparation complete!")
    print("📂 Artifacts saved in:", SAVE_DIR)
