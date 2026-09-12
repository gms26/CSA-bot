import pandas as pd


INPUT_FILE = "processed/semantic_retrieval_candidates_100.csv"
OUTPUT_FILE = "processed/semantic_retrieval_labeling_100.csv"


def main():
    print("=" * 70)
    print("PREPARE RETRIEVAL LABELING FILE")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"Candidates loaded: {len(df):,}")

    # Add an empty human relevance label.
    # 1 = relevant
    # 0 = not relevant
    df["relevance"] = ""

    # Add a place for an optional short reason.
    df["label_reason"] = ""

    # Put the most useful columns first.
    columns = [
        "eval_id",
        "query_thread_id",
        "query",
        "rank",
        "similarity",
        "retrieved_thread_id",
        "retrieved_customer_message",
        "retrieved_support_response",
        "relevance",
        "label_reason",
    ]

    df = df[columns]

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Saved to: {OUTPUT_FILE}")

    print()
    print("Columns:")
    print(df.columns.tolist())

    print()
    print("Rows per query:")
    print(df.groupby("eval_id").size().value_counts().sort_index())

    print()
    print("Labeling rule:")
    print("  relevance = 1 -> historically relevant to the query")
    print("  relevance = 0 -> not relevant to the query")
    print("  label_reason -> optional short explanation")


if __name__ == "__main__":
    main()