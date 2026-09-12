import pandas as pd


INPUT_FILE = "processed/apple_support_response_pairs_clean.csv"
OUTPUT_FILE = "processed/retrieval_eval_500.csv"

SAMPLE_SIZE = 500


def main():
    print("=" * 70)
    print("CREATE RETRIEVAL EVALUATION SET")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"Total response pairs: {len(df):,}")

    # Sample fixed evaluation queries.
    eval_df = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=42
    ).reset_index(drop=True)

    # Give each evaluation example a stable ID.
    eval_df.insert(
        0,
        "eval_id",
        range(1, len(eval_df) + 1)
    )

    eval_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Evaluation examples:  {len(eval_df):,}")
    print(f"Saved to:             {OUTPUT_FILE}")

    print()
    print("Columns:")
    print(eval_df.columns.tolist())

    print()
    print("First 5 examples:")
    print(
        eval_df[
            ["eval_id", "thread_id", "customer_message", "support_response"]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()