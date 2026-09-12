import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


HISTORY_FILE = "processed/apple_support_response_pairs_clean.csv"
EVAL_FILE = "processed/retrieval_eval_500.csv"
OUTPUT_FILE = "processed/semantic_retrieval_candidates_100.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
NUM_QUERIES = 100
TOP_K = 5


def main():
    print("=" * 70)
    print("CREATE SEMANTIC RETRIEVAL CANDIDATES")
    print("=" * 70)

    history = pd.read_csv(HISTORY_FILE)
    evaluation = pd.read_csv(EVAL_FILE).head(NUM_QUERIES)

    print(f"Historical pairs:     {len(history):,}")
    print(f"Evaluation queries:   {len(evaluation):,}")

    print()
    print(f"Loading model:        {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Creating historical embeddings...")

    history_embeddings = model.encode(
        history["customer_message"].fillna("").tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    results = []

    print()
    print("Retrieving top candidates...")

    for _, query_row in evaluation.iterrows():

        query_text = str(query_row["customer_message"])
        query_thread = query_row["thread_id"]
        eval_id = query_row["eval_id"]

        query_embedding = model.encode(
            [query_text],
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            query_embedding,
            history_embeddings
        )[0]

        # Exclude the original thread to prevent direct leakage.
        valid_indices = [
            i
            for i in range(len(history))
            if history.iloc[i]["thread_id"] != query_thread
        ]

        valid_scores = scores[valid_indices]

        top_positions = valid_scores.argsort()[::-1][:TOP_K]

        for rank, position in enumerate(top_positions, start=1):

            history_index = valid_indices[position]
            row = history.iloc[history_index]

            results.append({
                "eval_id": eval_id,
                "query_thread_id": query_thread,
                "query": query_text,
                "rank": rank,
                "similarity": round(float(scores[history_index]), 4),
                "retrieved_thread_id": row["thread_id"],
                "retrieved_customer_message": row["customer_message"],
                "retrieved_support_response": row["support_response"]
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Queries evaluated:    {NUM_QUERIES}")
    print(f"Candidates generated: {len(results_df):,}")
    print(f"Saved to:              {OUTPUT_FILE}")

    print()
    print("First query:")
    print("-" * 70)

    first_query = results_df[
        results_df["eval_id"] == results_df["eval_id"].iloc[0]
    ]

    print(
        first_query[
            [
                "rank",
                "similarity",
                "query",
                "retrieved_customer_message",
                "retrieved_support_response"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()