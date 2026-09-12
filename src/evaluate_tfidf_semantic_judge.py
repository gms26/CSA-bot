import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score


INPUT_FILE = "processed/tfidf_same_queries_candidates_100.csv"
OUTPUT_FILE = "processed/tfidf_semantic_judge_100.csv"
SUMMARY_FILE = "processed/tfidf_semantic_judge_summary.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

# Semantic similarity threshold.
# This is used only as an automatic secondary judge,
# not as a human/gold relevance label.
THRESHOLD = 0.60


def main():
    df = pd.read_csv(INPUT_FILE)

    model = SentenceTransformer(MODEL_NAME)

    queries = df["query"].astype(str).tolist()
    candidates = df["retrieved_customer_message"].astype(str).tolist()

    query_embeddings = model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    candidate_embeddings = model.encode(
        candidates,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    similarities = np.sum(
        query_embeddings * candidate_embeddings,
        axis=1
    )

    df["semantic_similarity"] = similarities
    df["semantic_relevant"] = (
        df["semantic_similarity"] >= THRESHOLD
    ).astype(int)

    df.to_csv(OUTPUT_FILE, index=False)

    results = []

    for eval_id, group in df.groupby("eval_id"):
        group = group.sort_values("rank")

        relevant = group["semantic_relevant"].tolist()
        similarities = group["semantic_similarity"].tolist()

        # Recall@K
        recall_1 = int(any(relevant[:1]))
        recall_3 = int(any(relevant[:3]))
        recall_5 = int(any(relevant[:5]))

        # Precision@5
        precision_5 = sum(relevant[:5]) / 5

        # MRR
        reciprocal_rank = 0.0
        for rank, value in enumerate(relevant, start=1):
            if value == 1:
                reciprocal_rank = 1.0 / rank
                break

        results.append({
            "eval_id": eval_id,
            "recall_at_1": recall_1,
            "recall_at_3": recall_3,
            "recall_at_5": recall_5,
            "precision_at_5": precision_5,
            "mrr": reciprocal_rank,
            "top1_similarity": similarities[0]
        })

    metrics = pd.DataFrame(results)

    summary = pd.DataFrame([{
        "queries": len(metrics),
        "semantic_judge_threshold": THRESHOLD,
        "recall_at_1": metrics["recall_at_1"].mean(),
        "recall_at_3": metrics["recall_at_3"].mean(),
        "recall_at_5": metrics["recall_at_5"].mean(),
        "precision_at_5": metrics["precision_at_5"].mean(),
        "mrr": metrics["mrr"].mean()
    }])

    metrics.to_csv(
        "processed/tfidf_semantic_judge_per_query.csv",
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print("\nTF-IDF automatic semantic-judge evaluation")
    print("-------------------------------------------")
    print(f"Queries:       {len(metrics)}")
    print(f"Threshold:     {THRESHOLD}")
    print(f"Recall@1:      {summary.iloc[0]['recall_at_1']:.4f}")
    print(f"Recall@3:      {summary.iloc[0]['recall_at_3']:.4f}")
    print(f"Recall@5:      {summary.iloc[0]['recall_at_5']:.4f}")
    print(f"Precision@5:   {summary.iloc[0]['precision_at_5']:.4f}")
    print(f"MRR:           {summary.iloc[0]['mrr']:.4f}")

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Saved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()