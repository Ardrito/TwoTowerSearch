import random
from datasets import load_dataset
import numpy as np


def extract_query_doc_pairs(split_data):
    """
    Extracts query-document pairs from the split data.
    Tracks how many positives were selected vs. fallback.
    """
    query_doc_pairs = []
    selected_count = 0
    fallback_count = 0
    total = len(split_data)

    for i in range(total):
        item = split_data[i]
        query_id = item['query_id']
        query_text = item['query']
        passages = item['passages']['passage_text']
        is_selected = item['passages'].get('is_selected', [0] * len(passages))

        positive_docs = [passages[j]
                         for j, selected in enumerate(is_selected) if selected == 1]

        if positive_docs:
            selected_count += 1
        elif passages:
            # Fallback to first passage
            positive_docs = [passages[0]]
            fallback_count += 1

        # Get passages and their relevance indicators
        passages = item['passages']['passage_text']
        is_selected = item['passages']['is_selected']

        positive_docs = []
        for j, selected in enumerate(is_selected):
            if selected == 1:
                positive_docs.append(passages[j])

        if positive_docs:
            query_doc_pairs.append({
                'query_id': query_id,
                'query': query_text,
                'positive_docs': positive_docs,
                'all_passages': passages
            })

        # Optional: print progress every 10k
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} / {total}")

    print(f"\n✅ Total query-doc pairs: {len(query_doc_pairs)}")
    print(f"   📌 With is_selected == 1: {selected_count}")
    print(f"   🧷 With fallback to first passage: {fallback_count}")
    return query_doc_pairs


def build_doc_table(query_doc_pairs):
    doc_table = {}         # {doc_id: doc_text}
    doc_to_id = {}         # {doc_text: doc_id}
    doc_id_counter = 0

    for pair in query_doc_pairs:
        for passage in pair['all_passages']:
            if passage not in doc_to_id:
                doc_id = f"d{doc_id_counter:06d}"
                doc_to_id[passage] = doc_id
                doc_table[doc_id] = passage
                doc_id_counter += 1
    return doc_table, doc_to_id


def generate_triples(query_doc_pairs, doc_to_id, num_negatives=1, negative_pool_size=100000):
    triples = []
    all_docs = list(doc_to_id.keys())
    negative_pool = random.sample(
        all_docs, min(negative_pool_size, len(all_docs)))

    for i, pair in enumerate(query_doc_pairs):
        query = pair['query']
        pos_docs = pair['positive_docs']

        for pos_doc in pos_docs:
            neg_docs = random.sample(negative_pool, num_negatives)
            for neg_doc in neg_docs:
                triples.append({
                    'query': query,
                    'pos_doc': pos_doc,
                    'neg_doc': neg_doc
                })

        if i % 5000 == 0 and i > 0:
            print(f"📊 Processed {i}/{len(query_doc_pairs)} queries...")

    return triples


# Testing functions

def test_extraction():
    # Load a small sample of the dataset
    print("Loading dataset...")
    dataset = load_dataset("ms_marco", "v1.1")

    # Use a small sample for quick testing
    sample_size = 100
    train_sample = dataset['train'].select(
        range(min(sample_size, len(dataset['train']))))

    print(f"Processing {len(train_sample)} samples from training data...")

    # Process the sample
    pairs = extract_query_doc_pairs(train_sample)

    print(
        f"Extracted {len(pairs)} query-document pairs with positive documents")

    # Print sample of results for verification
    if pairs:
        # Select a random pair to display
        sample_pair = random.choice(pairs)

        print("\nSample Query-Document Pair:")
        print(f"Query ID: {sample_pair['query_id']}")
        print(f"Query: {sample_pair['query']}")
        print(
            f"\nNumber of positive documents: {len(sample_pair['positive_docs'])}")

        if sample_pair['positive_docs']:
            print(
                f"\nFirst positive document: {sample_pair['positive_docs'][0][:200]}...")

        print(
            f"\nTotal documents for this query: {len(sample_pair['all_passages'])}")

        # Calculate ratio of positive to all documents
        pos_ratio = len(sample_pair['positive_docs']) / \
            len(sample_pair['all_passages'])
        print(f"Ratio of positive to all documents: {pos_ratio:.2f}")

    else:
        print("No query-document pairs found with positive documents in the sample.")

    return pairs


def find_specific_query(dataset, query_id):
    """
    Find and display detailed information about a specific query ID.

    Args:
        dataset: The dataset to search in
        query_id: The query ID to look for
    """
    print(f"Searching for query ID: {query_id}")

    # Search through training split
    found = False
    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue

        split_data = dataset[split_name]

        for i in range(len(split_data)):
            item = split_data[i]
            if item['query_id'] == query_id:
                found = True
                print(
                    f"Found query ID {query_id} in {split_name} split at index {i}")
                print(f"Query: {item['query']}")

                # Get passages and their relevance indicators
                passages = item['passages']['passage_text']
                is_selected = item['passages']['is_selected']

                print(f"Total passages: {len(passages)}")

                # Show which passages are marked as relevant
                relevant_indices = [j for j, selected in enumerate(
                    is_selected) if selected == 1]
                print(f"Relevant passage indices: {relevant_indices}")

                # Display relevant passages
                print("\nRelevant passages:")
                for j in relevant_indices:
                    print(f"Passage {j}: {passages[j]}")

                # Display a sample of non-relevant passages
                non_relevant = [j for j, selected in enumerate(
                    is_selected) if selected == 0]
                print("\nSample of non-relevant passages:")
                for j in non_relevant[:3]:  # Show first 3 non-relevant passages
                    print(f"Passage {j}: {passages[j]}")

                # Calculate ratio
                pos_ratio = len(relevant_indices) / len(passages)
                print(
                    f"\nRatio of positive to all documents: {pos_ratio:.2f} ({len(relevant_indices)}/{len(passages)})")

                break

        if found:
            break

    if not found:
        print(f"Query ID {query_id} not found in the dataset.")


if __name__ == "__main__":

    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("ms_marco", "v1.1")

    # Find specific query
    # find_specific_query(dataset, 19798)  # Example query ID 19798

    # Test preprocessing pipeline
    print("Testing extraction...")
    pairs = test_extraction()
    # Test generating triples
    print("Generating triples...")
    triples = generate_triples(pairs)
    print(f"Generated {len(triples)} triples.")
    if triples:
        print("\nSample triple:")
        print(triples[0])
