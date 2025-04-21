import random
from datasets import load_dataset


def extract_query_doc_pairs(split_data):
    """
    Extracts query-document pairs from the split data.

    Args:
        split_data: A dataset split (train, validation, or test)

    Returns:
        list: Query-document pairs with positive documents
    """
    query_doc_pairs = []

    for i in range(len(split_data)):
        item = split_data[i]
        query_id = item['query_id']
        query_text = item['query']

    # Get passages and their relevance indicators
    passages = item['passages']['passage_text']
    is_selected = item['passages']['is_selected']

    # Find positive (relevant) documents

    positive_docs = []
    for j, selected in enumerate(is_selected):
        if selected == 1:
            positive_docs.append(passages[j])

    # If there are positive documents, add to the pairs
    if positive_docs:
        query_doc_pairs.append({
            'query_id': query_id,
            'query': query_text,
            'positive_docs': positive_docs,
            'all_passages': passages
        })

    return query_doc_pairs


def generate_triples(query_doc_pairs, num_negatives=1):
    """
    Generates triples from query-document pairs.

    Args:
        query_doc_pairs: List of query-document pairs with positive docs
        num_negatives: Number of negative documents to sample

    Returns:
        list: List of triples (query, positive_doc, negative_doc)
    """
    triples = []

    for pair in query_doc_pairs:
        query = pair['query']
        query_id = pair['query_id']
        positive_docs = pair['positive_docs']
        all_passages = pair['all_passages']

        # For each positive document
        for pos_doc in positive_docs:
            # Create a list of passages that are not positive for this query
            negative_candidates = [
                p for p in all_passages if p not in positive_docs]
            # If no negative candidates, skip this query
            if len(negative_candidates) < num_negatives:
                continue
        # Sample negative documents
            neg_docs = random.sample(negative_candidates, min(
                num_negatives, len(negative_candidates)))
            # Create triples for each negative document
            for neg_doc in neg_docs:
                triples.append({
                    'query_id': query_id,
                    'query': query,
                    'pos_doc': pos_doc,
                    'neg_doc': neg_doc
                })
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
