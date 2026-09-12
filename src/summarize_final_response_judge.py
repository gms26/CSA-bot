import pandas as pd


INPUT_FILE = "processed/final_response_judge_results_20.csv"


def main():
    print("=" * 70)
    print("CSA — FINAL RESPONSE JUDGE SUMMARY")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    metrics = [
        "relevant",
        "grounded",
        "safe",
        "escalation_correct",
    ]

    print(f"Examples evaluated: {len(df)}")
    print(
        f"Eval ID range: "
        f"{df['eval_id'].min()}–{df['eval_id'].max()}"
    )
    print()

    for metric in metrics:
        values = pd.to_numeric(df[metric], errors="coerce")
        valid = values.dropna()

        score = valid.mean()
        passed = int(valid.sum())
        total = len(valid)
        failed = total - passed

        print(
            f"{metric:22s}: "
            f"{score:.2%} "
            f"({passed}/{total} passed, {failed} failed)"
        )

    print()
    print("-" * 70)
    print("FAILURE COUNTS")
    print("-" * 70)

    for metric in metrics:
        values = pd.to_numeric(df[metric], errors="coerce")
        failures = df[values == 0]

        print(f"\n{metric.upper()} failures: {len(failures)}")

        for _, row in failures.iterrows():
            print(
                f"  Eval {row['eval_id']}: "
                f"{row['customer_message'][:100]}"
            )
            print(
                f"    Response: "
                f"{row['response'][:150]}"
            )
            print(
                f"    Reason: "
                f"{row['judge_reason']}"
            )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
    