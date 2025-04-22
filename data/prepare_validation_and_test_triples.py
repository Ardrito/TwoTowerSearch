import os
import json
from datasets import load_dataset
from preprocessing import extract_query_doc_pairs, build_doc_table, generate_triples

SAVE_DIR = "saved_artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {path}")

def process_split(split_name, split_data, save_name):
    print(f"\n📦 Processing split: {split_name}")
    
    print("🔍 Extracting query-document pairs...")
    pairs = extract_query_doc_pairs(split_data)
    print(f"✅ Extracted {len(pairs)} {split_name} pairs.")

    print("📇 Building document index...")
    doc_table, doc_to_id = build_doc_table(pairs)

    print("🔁 Generating triples...")
    triples = generate_triples(pairs, doc_to_id, num_negatives=1)
    print(f"✅ Generated {len(triples)} {split_name} triples.")

    save_path = os.path.join(SAVE_DIR, f"triples_{save_name}.json")
    save_json(triples, save_path)

if __name__ == "__main__":
    print("📥 Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")

    if "validation" in dataset:
        process_split("validation", dataset["validation"], "validation")
    else:
        print("⚠️ Validation split not found!")

    if "test" in dataset:
        process_split("test", dataset["test"], "test")
    else:
        print("⚠️ Test split not found!")
