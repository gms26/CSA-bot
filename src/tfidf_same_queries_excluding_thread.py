import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


EVAL_FILE = "processed/retrieval_eval_500.csv"
HISTORY_FILE = "processed/apple_support_response_pairs_clean.csv"
OUTPUT_FILE = "processed/tfidf_same_queries_excluding_thread_100.csv"


def main():
    eval_df = pd.read_csv(EVAL_FILE)
    history = pd.read_csv(HISTORY_FILE)

    # Use evaluation IDs 1-100.
    eval_queries = (
        eval_df[eval_df["eval_id"].between(1, 100)]
        [["eval_id", "customer_message", "thread_id"]]
        .drop_duplicates("eval_id")
        .sort_values("eval_id")
    )

    print("Historical pairs:", len(history))
    print("Evaluation queries:", len(eval_queries))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    history_text = (
        history["customer_message"]
        .fillna("")
        .astype(str)
    )

    history_matrix = vectorizer.fit_transform(history_text)

    results = []

    for _, query_row in eval_queries.iterrows():
        eval_id = query_row["eval_id"]
        query = str(query_row["customer_message"])
        original_thread = query_row["thread_id"]

        query_vector = vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            history_matrix
        ).ravel()

        # Exclude the complete original thread.
        valid_mask = (
            history["thread_id"].astype(str)
            != str(original_thread)
        )

        scores[~valid_mask.to_numpy()] = -1

        top_indices = scores.argsort()[::-1][:5]

        for rank, idx in enumerate(top_indices, start=1):
            row = history.iloc[idx]

            results.append({
                "eval_id": eval_id,
                "query": query,
                "original_thread_id": original_thread,
                "rank": rank,
                "similarity": scores[idx],
                "retrieved_thread_id": row["thread_id"],
                "retrieved_customer_message": row["customer_message"],
                "retrieved_support_response": row["support_response"]
            })

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    leakage = (
        result_df["original_thread_id"].astype(str)
        == result_df["retrieved_thread_id"].astype(str)
    ).sum()

    print("\nTF-IDF retrieval with thread exclusion")
    print("---------------------------------------")
    print("Queries:", result_df["eval_id"].nunique())
    print("Candidates:", len(result_df))
    print("Original-thread leakage:", leakage)

    print("\nFirst 10 queries:\n")

    for eval_id, group in result_df.groupby("eval_id"):
        if eval_id > 10:
            break

        print("=" * 80)
        print("Eval ID:", eval_id)
        print("Query:", group.iloc[0]["query"])

        for _, row in group.iterrows():
            print(
                f"\nRank {row['rank']} "
                f"(similarity={row['similarity']:.4f})"
            )
            print(row["retrieved_customer_message"])

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()