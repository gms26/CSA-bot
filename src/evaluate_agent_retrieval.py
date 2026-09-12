import pandas as pd


INPUT_FILE = "processed/semantic_retrieval_labeling_100.csv"
OUTPUT_FILE = "processed/agent_retrieval_evaluation.csv"


def main():
    print("=" * 70)
    print("CSA AGENT — RETRIEVAL EVALUATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    df["relevance"] = df["relevance"].astype(int)

    results = []

    for eval_id, group in df.groupby("eval_id"):
        group = group.sort_values("rank")

        relevant = group["relevance"].tolist()

        first_relevant_rank = None

        for rank, value in enumerate(relevant, start=1):
            if value == 1:
                first_relevant_rank = rank
                break

        results.append(
            {
                "eval_id": eval_id,
                "recall_at_1": int(any(relevant[:1])),
                "recall_at_3": int(any(relevant[:3])),
                "recall_at_5": int(any(relevant[:5])),
                "precision_at_5": sum(relevant[:5]) / 5,
                "first_relevant_rank": first_relevant_rank,
            }
        )

    metrics = pd.DataFrame(results)

    recall_at_1 = metrics["recall_at_1"].mean()
    recall_at_3 = metrics["recall_at_3"].mean()
    recall_at_5 = metrics["recall_at_5"].mean()
    precision_at_5 = metrics["precision_at_5"].mean()

    reciprocal_ranks = metrics["first_relevant_rank"].apply(
        lambda rank: 0.0 if pd.isna(rank) else 1.0 / rank
    )

    mrr = reciprocal_ranks.mean()

    print()
    print(f"Queries:       {len(metrics)}")
    print(f"Recall@1:      {recall_at_1:.4f}")
    print(f"Recall@3:      {recall_at_3:.4f}")
    print(f"Recall@5:      {recall_at_5:.4f}")
    print(f"Precision@5:   {precision_at_5:.4f}")
    print(f"MRR:           {mrr:.4f}")

    metrics.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()