import pandas as pd


INPUT_FILE = "processed/response_evaluation_20.csv"
OUTPUT_FILE = "processed/response_judge_input_20.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    judge_df = df[
        [
            "eval_id",
            "customer_message",
            "predicted_intent",
            "response",
            "escalate",
            "escalation_reason",
            "top_retrieval_score",
        ]
    ].copy()

    judge_df["grounded"] = ""
    judge_df["relevant"] = ""
    judge_df["safe"] = ""
    judge_df["escalation_correct"] = ""
    judge_df["judge_reason"] = ""

    judge_df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 70)
    print("CSA — RESPONSE JUDGE INPUT")
    print("=" * 70)
    print(f"Examples: {len(judge_df)}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()