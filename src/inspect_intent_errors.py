import csv
from collections import Counter


INPUT_FILE = "processed/intent_classifier_predictions.csv"

MAX_ERRORS = 50


with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)


errors = [
    row
    for row in rows
    if row["correct"].lower() == "false"
]


print("=" * 70)
print("INTENT CLASSIFIER ERROR ANALYSIS")
print("=" * 70)

print(f"Total examples: {len(rows)}")
print(f"Errors: {len(errors)}")
print(f"Correct: {len(rows) - len(errors)}")


error_pairs = Counter(
    (row["true_intent"], row["predicted_intent"])
    for row in errors
)


print("\nMost common confusion pairs:")

for (true_intent, predicted_intent), count in error_pairs.most_common(15):
    print(
        f"{true_intent} -> {predicted_intent}: {count}"
    )


print("\n" + "=" * 70)
print("MISCLASSIFIED EXAMPLES")
print("=" * 70)


for i, row in enumerate(errors[:MAX_ERRORS], start=1):

    print("\n" + "-" * 70)
    print(f"ERROR #{i}")
    print(f"Tweet ID: {row['tweet_id']}")
    print(f"True intent: {row['true_intent']}")
    print(f"Predicted: {row['predicted_intent']}")
    print(f"Message: {row['text']}")