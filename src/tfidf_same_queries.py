import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_PAIRS = "processed/apple_support_response_pairs_clean.csv"
INPUT_EVAL = "processed/semantic_retrieval_labeling_100.csv"
OUTPUT_FILE = "processed/tfidf_same_queries_candidates_100.csv"


# Load historical response pairs
pairs = pd.read_csv(INPUT_PAIRS)

# Load the semantic evaluation file
eval_df = pd.read_csv(INPUT_EVAL)

# Get exactly one row for each evaluation query
queries = (
    eval_df
    .sort_values(["eval_id", "rank"])
    .groupby("eval_id", as_index=False)
    .first()
)

print(f"Historical pairs: {len(pairs)}")
print(f"Evaluation queries: {len(queries)}")


# Build TF-IDF index over historical customer messages
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

history_matrix = vectorizer.fit_transform(
    pairs["customer_message"].fillna("")
)


results = []

for _, query_row in queries.iterrows():

    eval_id = query_row["eval_id"]
    query = str(query_row["query"])

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        history_matrix
    )[0]

    ranked_indices = scores.argsort()[::-1]

    rank = 0

    for idx in ranked_indices:

        rank += 1

        if rank > 5:
            break

        row = pairs.iloc[idx]

        results.append({
            "eval_id": eval_id,
            "query": query,
            "rank": rank,
            "score": float(scores[idx]),
            "retrieved_thread_id": row["thread_id"],
            "retrieved_customer_message": row["customer_message"],
            "retrieved_support_response": row["support_response"],
        })


results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 60)
print("TF-IDF SAME-QUERY RETRIEVAL")
print("=" * 60)
print(f"Queries: {results_df['eval_id'].nunique()}")
print(f"Candidates: {len(results_df)}")
print(f"Saved: {OUTPUT_FILE}")
print("=" * 60)