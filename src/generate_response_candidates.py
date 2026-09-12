import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


HISTORY_FILE = "processed/apple_support_response_pairs_clean.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

OUTPUT_FILE = "processed/response_generation_candidates.csv"


def main():
    history = pd.read_csv(HISTORY_FILE)

    model = SentenceTransformer(MODEL_NAME)

    print("Historical response pairs:", len(history))

    # Use a small set of representative customer messages.
    test_queries = [
        "My iPhone battery is draining very quickly after the update.",
        "My iPhone keeps freezing after updating to iOS 11.",
        "I cannot connect to Wi-Fi on my iPhone.",
        "I cannot download apps from the App Store.",
        "My Apple ID password is not working.",
        "My iPhone screen is not responding."
    ]

    customer_messages = (
        history["customer_message"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("Encoding historical customer messages...")

    history_embeddings = model.encode(
        customer_messages,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    query_embeddings = model.encode(
        test_queries,
        normalize_embeddings=True
    )

    results = []

    for query, query_embedding in zip(
        test_queries,
        query_embeddings
    ):
        scores = cosine_similarity(
            [query_embedding],
            history_embeddings
        )[0]

        top_indices = scores.argsort()[::-1][:5]

        for rank, idx in enumerate(top_indices, start=1):
            row = history.iloc[idx]

            results.append({
                "query": query,
                "rank": rank,
                "similarity": scores[idx],
                "historical_customer_message":
                    row["customer_message"],
                "historical_support_response":
                    row["support_response"]
            })

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nResponse generation candidates")
    print("------------------------------")

    for query, group in result_df.groupby("query"):
        print("\n" + "=" * 80)
        print("CUSTOMER:")
        print(query)

        for _, row in group.iterrows():
            print(
                f"\nRank {row['rank']} "
                f"(similarity={row['similarity']:.4f})"
            )

            print("Historical customer:")
            print(row["historical_customer_message"])

            print("Historical AppleSupport response:")
            print(row["historical_support_response"])

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()