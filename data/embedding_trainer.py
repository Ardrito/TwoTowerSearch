import os
import json
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from datasets import load_dataset
from preprocessing import extract_query_doc_pairs, generate_triples
from tokenization import tokenize_texts


def train_word2vec(sentences, vector_size=300, window=5, min_count=2):
    """
    Trains a Word2Vec model using CBOW.

    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of word vectors
        window: Context window size
        min_count: Minimum word frequency

    Returns:
        model: Trained Word2Vec model
    """
    print("🧠 Training Word2Vec model (CBOW)...")
    model = Word2Vec(sentences, vector_size=vector_size,
                     window=window, min_count=min_count, sg=0)
    return model


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {filepath}")


if __name__ == "__main__":
    # 1. Load a sample of the dataset
    print("📦 Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")
    # Feel free to increase for full run
    sample = dataset["train"].select(range(1000))

    # 2. Extract and generate triples
    print("🔍 Extracting query-document pairs and generating triples...")
    pairs = extract_query_doc_pairs(sample)
    triples = generate_triples(pairs, num_negatives=1)

    # 3. Tokenize the text
    print("✂️ Tokenizing text fields...")
    tokenized_sentences = tokenize_texts(triples)

    # 4. Train Word2Vec model
    model = train_word2vec(tokenized_sentences)

    # 5. Save everything
    save_dir = "saved_artifacts"
    os.makedirs(save_dir, exist_ok=True)

    print("💾 Saving Word2Vec model...")
    model.save(os.path.join(save_dir, "msmarco_word2vec_cbow.model"))

    print("💾 Saving word vectors in .vec format...")
    model.wv.save_word2vec_format(os.path.join(
        save_dir, "msmarco_word_vectors.vec"), binary=False)

    print("💾 Saving tokenized sentences...")
    save_json(tokenized_sentences, os.path.join(
        save_dir, "tokenized_sentences.json"))

    print("💾 Saving generated triples...")
    save_json(triples, os.path.join(save_dir, "triples.json"))

    # 6. Optional: Check similarity
    print("\n🧪 Sample word similarity:")
    try:
        print(model.wv.most_similar("tower", topn=5))
    except KeyError:
        print("Word 'tower' not in vocabulary.")

    print("\n✅ All done! Artifacts saved in:", save_dir)
