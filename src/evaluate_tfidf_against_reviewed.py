import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SEMANTIC_LABELS = "processed/semantic_retrieval_labeling_100.csv"
TFIDF_CANDIDATES = "processed/tfidf_same_queries_candidates_100.csv"

OUTPUT = "processed/tfidf_against_reviewed_100.csv"


def main():
    reviewed = pd.read_csv(SEMANTIC_LABELS)
    tfidf = pd.read_csv(TFIDF_CANDIDATES)

    # Only candidates that were manually judged relevant
    relevant = reviewed[reviewed["relevance"] == 1].copy()

    # Unique relevant historical messages
    relevant_messages = (
        relevant["retrieved_customer_message"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    results = []

    for eval_id, group in tfidf.groupby("eval_id"):
        query = str(group.iloc[0]["query"])

        candidates = group.sort_values("rank")[
            "retrieved_customer_message"
        ].astype(str).tolist()

        # Compare TF-IDF retrieved messages against the
        # reviewed relevant historical messages.
        if not relevant_messages:
            best_similarity = [0.0] * len(candidates)
        else:
            texts = [query] + candidates + relevant_messages

            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1
            )

            matrix = vectorizer.fit_transform(texts)

            candidate_matrix = matrix[
                1:1 + len(candidates)
            ]

            relevant_matrix = matrix[
                1 + len(candidates):
            ]

            similarities = cosine_similarity(
                candidate_matrix,
                relevant_matrix
            )

            best_similarity = similarities.max(axis=1)

        # A retrieved candidate is considered to match the
        # reviewed relevance pool if lexical similarity is high.
        MATCH_THRESHOLD = 0.50

        matched = [
            int(score >= MATCH_THRESHOLD)
            for score in best_similarity
        ]

        results.append({
            "eval_id": eval_id,
            "query": query,
            "tfidf_rank1_matches_reviewed": matched[0],
            "tfidf_rank3_matches_reviewed": int(any(matched[:3])),
            "tfidf_rank5_matches_reviewed": int(any(matched[:5])),
            "best_rank1_similarity": best_similarity[0],
            "best_rank5_similarity": max(best_similarity[:5])
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT, index=False)

    print("\nTF-IDF vs reviewed semantic relevance pool")
    print("-----------------------------------------")
    print("Queries:", len(results_df))
    print(
        "Rank-1 coverage:",
        f"{results_df['tfidf_rank1_matches_reviewed'].mean():.4f}"
    )
    print(
        "Rank-3 coverage:",
        f"{results_df['tfidf_rank3_matches_reviewed'].mean():.4f}"
    )
    print(
        "Rank-5 coverage:",
        f"{results_df['tfidf_rank5_matches_reviewed'].mean():.4f}"
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()