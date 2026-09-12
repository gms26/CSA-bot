import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PAIR_FILE = "processed/apple_support_response_pairs_clean.csv"
EVAL_FILE = "processed/apple_customer_sample_500_labeled.csv"

TOP_K = 5


print("Loading historical response pairs...")

pairs = []

with open(PAIR_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        pairs.append(row)

print(f"Historical pairs: {len(pairs)}")


print("Loading labelled evaluation examples...")

evaluation_rows = []

with open(EVAL_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        evaluation_rows.append(row)

print(f"Evaluation examples: {len(evaluation_rows)}")


documents = [
    row["customer_message"]
    for row in pairs
]


print("Building TF-IDF index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

document_vectors = vectorizer.fit_transform(documents)

print("Index ready.")


def retrieve(query, top_k=TOP_K):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices:

        result = pairs[index]

        # Avoid returning an exact same customer message.
        if result["customer_message"].strip().lower() == query.strip().lower():
            continue

        results.append({
            "score": float(similarities[index]),
            "customer_message": result["customer_message"],
            "support_response": result["support_response"],
        })

        if len(results) == top_k:
            break

    return results


# ---------------------------------------------------------
# Evaluate whether retrieved examples share the same intent
# ---------------------------------------------------------

intent_match_at_1 = 0
intent_match_at_5 = 0

# The historical response-pair dataset does not yet have
# manually assigned intent labels, so we cannot calculate
# intent-based retrieval accuracy yet.
#
# For now, we measure retrieval coverage and save the
# retrieved examples for later human/LLM evaluation.

OUTPUT_FILE = "processed/tfidf_retrieval_evaluation.csv"


with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:

    fieldnames = [
        "eval_id",
        "intent",
        "query",
        "rank",
        "score",
        "retrieved_customer_message",
        "retrieved_support_response",
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for row in evaluation_rows:

        eval_id = row["tweet_id"]
        intent = row["intent"]
        query = row["text"]

        results = retrieve(query)

        for rank, result in enumerate(results, start=1):

            writer.writerow({
                "eval_id": eval_id,
                "intent": intent,
                "query": query,
                "rank": rank,
                "score": f"{result['score']:.6f}",
                "retrieved_customer_message": result["customer_message"],
                "retrieved_support_response": result["support_response"],
            })


print("=" * 60)
print("TF-IDF RETRIEVAL EVALUATION")
print("=" * 60)
print(f"Evaluation queries: {len(evaluation_rows)}")
print(f"Top K: {TOP_K}")
print(f"Output: {OUTPUT_FILE}")