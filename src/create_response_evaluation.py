import pandas as pd
import sys
sys.path.insert(0, ".")

from src.customer_support_agent import CustomerSupportAgent

INPUT_FILE = "processed/retrieval_eval_500.csv"
OUTPUT_FILE = "processed/response_evaluation_20.csv"


def main():
    print("=" * 70)
    print("CSA — RESPONSE EVALUATION DATASET")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    # Use the first 20 evaluation examples for the initial response evaluation.
    eval_df = df.head(20).copy()

    agent = CustomerSupportAgent()

    results = []

    for index, row in eval_df.iterrows():
        print()
        print(f"Generating response {index + 1}/{len(eval_df)}")

        result = agent.handle(row["customer_message"])

        results.append({
            "eval_id": row["eval_id"],
            "customer_message": row["customer_message"],
            "predicted_intent": result["intent"],
            "response": result["response"],
            "escalate": result["escalate"],
            "escalation_reason": result["escalation_reason"],
            "top_retrieval_score": (
                result["retrieved_examples"][0]["similarity"]
                if result["retrieved_examples"]
                else 0.0
            ),
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 70)
    print(f"Generated {len(output_df)} response evaluations")
    print(f"Saved to {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()