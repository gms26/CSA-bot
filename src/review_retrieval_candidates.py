import argparse
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


INPUT_FILE = "processed/semantic_retrieval_labeling_100.csv"

NUM_QUERIES = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("RETRIEVAL CANDIDATE REVIEW")
    print("=" * 80)

    sorted_eval_ids = sorted(df["eval_id"].unique())
    if args.start is None and args.end is None:
        eval_ids = sorted_eval_ids[:NUM_QUERIES]
    else:
        start = args.start if args.start is not None else sorted_eval_ids[0]
        end = args.end if args.end is not None else sorted_eval_ids[-1]
        eval_ids = [eval_id for eval_id in sorted_eval_ids if start <= eval_id <= end]

    for eval_id in eval_ids:

        group = df[df["eval_id"] == eval_id]

        print()
        print("=" * 80)
        print(f"EVALUATION QUERY: {eval_id}")
        print("=" * 80)

        query = group.iloc[0]["query"]

        print(f"\nQUERY:\n{query}")

        for _, row in group.iterrows():

            print()
            print("-" * 80)
            print(
                f"RANK: {int(row['rank'])}   "
                f"SIMILARITY: {row['similarity']:.4f}"
            )

            print(f"\nRetrieved customer message:")
            print(row["retrieved_customer_message"])

            print(f"\nHistorical AppleSupport response:")
            print(row["retrieved_support_response"])

            print()
            print("Label:")
            print("  1 = Relevant")
            print("  0 = Not relevant")


if __name__ == "__main__":
    main()