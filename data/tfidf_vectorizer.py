import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load your raw texts
with open("saved_artifacts/raw_queries.json", encoding="utf-8") as f:
    queries = json.load(f)

with open("saved_artifacts/raw_docs.json", encoding="utf-8") as f:
    docs = json.load(f)

# Combine into a single corpus (can weigh later if needed)
corpus = queries + docs

# Build the TF-IDF vectorizer
print("🔎 Training TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    lowercase=True, stop_words='english', max_features=10000)
vectorizer.fit(corpus)

# Save the fitted vectorizer for reuse
with open("saved_artifacts/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Saved TF-IDF vectorizer with", len(vectorizer.vocabulary_), "tokens.")
