import sys
sys.path.insert(0, ".")

import pandas as pd

from src.customer_support_agent import CustomerSupportAgent


INPUT_FILE = "processed/retrieval_eval_500.csv"
OUTPUT_FILE = "processed/final_response_evaluation_20.csv"


def main():
    print("=" * 70)
    print("CSA — FINAL RESPONSE EVALUATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    # Eval IDs 1–20 were used during development.
    # IDs 21–40 are kept as a fresh evaluation set.
    eval_df = df[
        (df["eval_id"] >= 21)
        & (df["eval_id"] <= 40)
    ].copy()

    if len(eval_df) != 20:
        raise RuntimeError(
            f"Expected 20 evaluation examples, found {len(eval_df)}."
        )

    agent = CustomerSupportAgent()

    results = []

    for index, row in eval_df.iterrows():
        print()
        print(
            f"Generating final response "
            f"{index - eval_df.index[0] + 1}/{len(eval_df)}"
        )

        result = agent.handle(row["customer_message"])

        retrieved = result.get("retrieved_examples", [])

        evidence = []

        for rank, item in enumerate(retrieved, start=1):
            evidence.append(
                f"Rank {rank} | Similarity: {item['similarity']:.4f}\n"
                f"Customer: {item['customer_message']}\n"
                f"AppleSupport: {item['support_response']}"
            )

        results.append({
            "eval_id": row["eval_id"],
            "customer_message": row["customer_message"],
            "predicted_intent": result["intent"],
            "response": result["response"],
            "escalate": result["escalate"],
            "escalation_reason": result["escalation_reason"],
            "top_retrieval_score": (
                retrieved[0]["similarity"]
                if retrieved
                else 0.0
            ),
            "retrieved_evidence": "\n\n".join(evidence),
        })

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print(f"Final evaluation examples: {len(output_df)}")
    print(f"Eval ID range: {output_df['eval_id'].min()}–{output_df['eval_id'].max()}")
    print(f"Saved to {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()