import re
from preprocessing import extract_query_doc_pairs, generate_triples
from datasets import load_dataset

def simple_tokenize(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Split on whitespace
    return text.split()

def tokenize_texts(triples):
    all_texts = []
    for triple in triples:
        all_texts.append(simple_tokenize(triple['query']))
        all_texts.append(simple_tokenize(triple['pos_doc']))
        all_texts.append(simple_tokenize(triple['neg_doc']))
    return all_texts


if __name__ == "__main__":
    # Step 1: Load a small sample of the dataset
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")
    # Limit to 100 items for quick test
    sample = dataset['train'].select(range(100))

    # Step 2: Extract query-document pairs
    print("Extracting query-document pairs...")
    pairs = extract_query_doc_pairs(sample)

    # Step 3: Generate triples
    print("Generating query-positive-negative triples...")
    triples = generate_triples(pairs, num_negatives=1)

    # Step 4: Tokenize the triples
    print("Tokenizing text fields from triples...")
    tokenized = tokenize_texts(triples)

    # Step 5: Print a few samples
    print("\n🔍 Sample Tokenized Sentences:")
    for i, tokens in enumerate(tokenized[:5]):  # Print first 5 tokenized lists
        print(f"{i+1}: {tokens}")
