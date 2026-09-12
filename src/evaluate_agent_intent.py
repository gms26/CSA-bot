import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from intent_classifier import IntentClassifier


INPUT_FILE = "processed/intent_test.csv"
OUTPUT_FILE = "processed/agent_intent_evaluation.csv"


def main():
    print("=" * 70)
    print("CSA AGENT — INTENT EVALUATION")
    print("=" * 70)

    test_df = pd.read_csv(INPUT_FILE)

    classifier = IntentClassifier()

    predictions = []

    for _, row in test_df.iterrows():
        result = classifier.predict(row["text"])

        predictions.append(result["intent"])

    test_df["predicted_intent"] = predictions

    accuracy = accuracy_score(
        test_df["intent"],
        test_df["predicted_intent"],
    )

    macro_f1 = f1_score(
        test_df["intent"],
        test_df["predicted_intent"],
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        test_df["intent"],
        test_df["predicted_intent"],
        average="weighted",
        zero_division=0,
    )

    print()
    print(f"Examples:    {len(test_df)}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            test_df["intent"],
            test_df["predicted_intent"],
            zero_division=0,
        )
    )

    test_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved predictions to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()