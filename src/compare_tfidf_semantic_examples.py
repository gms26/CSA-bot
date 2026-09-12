import pandas as pd


TFIDF_FILE = "processed/tfidf_same_queries_candidates_100.csv"
SEMANTIC_FILE = "processed/semantic_retrieval_candidates_100.csv"

OUTPUT_FILE = "processed/tfidf_vs_semantic_examples.csv"


def main():
    tfidf = pd.read_csv(TFIDF_FILE)
    semantic = pd.read_csv(SEMANTIC_FILE)

    tfidf_top = (
        tfidf[tfidf["rank"] == 1][
            ["eval_id", "query", "retrieved_customer_message"]
        ]
        .rename(columns={
            "retrieved_customer_message": "tfidf_top1"
        })
    )

    semantic_top = (
        semantic[semantic["rank"] == 1][
            ["eval_id", "retrieved_customer_message"]
        ]
        .rename(columns={
            "retrieved_customer_message": "semantic_top1"
        })
    )

    comparison = tfidf_top.merge(
        semantic_top,
        on="eval_id",
        how="inner"
    )

    # Identify cases where the two methods selected
    # different top results.
    comparison["different_top1"] = (
        comparison["tfidf_top1"].str.lower().str.strip()
        != comparison["semantic_top1"].str.lower().str.strip()
    )

    comparison.to_csv(OUTPUT_FILE, index=False)

    print("\nTF-IDF vs Semantic Retrieval")
    print("---------------------------")
    print("Queries:", len(comparison))
    print(
        "Different Top-1 results:",
        comparison["different_top1"].sum()
    )

    print("\nFirst 20 differences:\n")

    differences = comparison[
        comparison["different_top1"]
    ].head(20)

    for _, row in differences.iterrows():
        print("=" * 80)
        print("Eval ID:", row["eval_id"])
        print("Query:", row["query"])
        print("\nTF-IDF:")
        print(row["tfidf_top1"])
        print("\nSemantic:")
        print(row["semantic_top1"])

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()