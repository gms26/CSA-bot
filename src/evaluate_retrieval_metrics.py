import pandas as pd


INPUT_FILE = "processed/semantic_retrieval_labeling_100.csv"
OUTPUT_FILE = "processed/retrieval_metrics_summary.csv"


df = pd.read_csv(INPUT_FILE)

# Make sure data is in rank order
df = df.sort_values(["eval_id", "rank"])

results = []

for eval_id, group in df.groupby("eval_id"):
    relevance = group["relevance"].astype(int).tolist()

    # Recall@K:
    # 1 if at least one relevant result appears in top K,
    # otherwise 0.
    r1 = int(any(relevance[:1]))
    r3 = int(any(relevance[:3]))
    r5 = int(any(relevance[:5]))

    # Precision@5:
    # proportion of the top 5 results that are relevant.
    p5 = sum(relevance[:5]) / 5

    # Reciprocal Rank:
    # 1 / position of the first relevant result.
    reciprocal_rank = 0.0

    for rank, rel in enumerate(relevance, start=1):
        if rel == 1:
            reciprocal_rank = 1.0 / rank
            break

    results.append({
        "eval_id": eval_id,
        "recall_at_1": r1,
        "recall_at_3": r3,
        "recall_at_5": r5,
        "precision_at_5": p5,
        "reciprocal_rank": reciprocal_rank,
    })


metrics_df = pd.DataFrame(results)

summary = {
    "num_queries": len(metrics_df),
    "recall_at_1": metrics_df["recall_at_1"].mean(),
    "recall_at_3": metrics_df["recall_at_3"].mean(),
    "recall_at_5": metrics_df["recall_at_5"].mean(),
    "precision_at_5": metrics_df["precision_at_5"].mean(),
    "mrr": metrics_df["reciprocal_rank"].mean(),
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("SEMANTIC RETRIEVAL EVALUATION")
print("=" * 60)

print(f"Queries:       {summary['num_queries']}")
print(f"Recall@1:      {summary['recall_at_1']:.4f}")
print(f"Recall@3:      {summary['recall_at_3']:.4f}")
print(f"Recall@5:      {summary['recall_at_5']:.4f}")
print(f"Precision@5:   {summary['precision_at_5']:.4f}")
print(f"MRR:           {summary['mrr']:.4f}")

print("=" * 60)
print(f"Saved: {OUTPUT_FILE}")