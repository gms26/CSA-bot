import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = "processed/apple_support_response_pairs_clean.csv"
TOP_K = 5


print("Loading response pairs...")

rows = []

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        rows.append(row)


print(f"Loaded {len(rows)} response pairs.")


customer_messages = [
    row["customer_message"]
    for row in rows
]


print("Building standard TF-IDF index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

document_vectors = vectorizer.fit_transform(customer_messages)

print("Retrieval system ready.")


def retrieve(query, top_k=TOP_K):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:

        results.append({
            "score": float(similarities[index]),
            "customer_message": rows[index]["customer_message"],
            "support_response": rows[index]["support_response"],
        })

    return results


# Test queries

test_queries = [
    "My iPhone battery is draining very quickly",
    "My Wi-Fi is not connecting",
    "The App Store will not update my apps",
]


for query in test_queries:

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = retrieve(query)

    for rank, result in enumerate(results, start=1):

        print(f"\n--- Result {rank} ---")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Customer: {result['customer_message']}")
        print(f"AppleSupport: {result['support_response']}")