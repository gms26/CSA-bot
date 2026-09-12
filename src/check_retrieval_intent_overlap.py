import pandas as pd


RETRIEVAL_FILE = "processed/retrieval_eval_500.csv"
INTENT_FILE = "processed/apple_customer_sample_500_labeled.csv"


def main():
    print("=" * 70)
    print("CHECK RETRIEVAL / INTENT EVALUATION OVERLAP")
    print("=" * 70)

    retrieval = pd.read_csv(RETRIEVAL_FILE)
    intent = pd.read_csv(INTENT_FILE)

    print(f"Retrieval evaluation examples: {len(retrieval):,}")
    print(f"Intent-labelled examples:      {len(intent):,}")

    retrieval_messages = set(
        retrieval["customer_message"]
        .fillna("")
        .str.strip()
        .str.lower()
    )

    intent_messages = set(
        intent["text"]
        .fillna("")
        .str.strip()
        .str.lower()
    )

    overlap = retrieval_messages.intersection(intent_messages)

    print()
    print(f"Exact message overlap:          {len(overlap):,}")

    if overlap:
        print()
        print("Overlapping examples:")
        print("-" * 70)

        matched = intent[
            intent["text"]
            .fillna("")
            .str.strip()
            .str.lower()
            .isin(overlap)
        ]

        print(
            matched[
                ["text", "intent"]
            ].head(20).to_string(index=False)
        )

    print()
    print("=" * 70)

    if len(overlap) == 0:
        print("No exact overlap.")
        print("Retrieval evaluation needs its own relevance labels.")
    else:
        print("Some evaluation queries already have intent labels.")
        print("We can use the overlap carefully for a limited analysis.")


if __name__ == "__main__":
    main()