import csv
from collections import Counter

INPUT_FILE = "processed/apple_customer_sample_500_labeled.csv"
OUTPUT_FILE = "processed/intent_baseline_results.csv"

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

true_labels = [row["intent"] for row in rows]

counts = Counter(true_labels)

majority_intent, majority_count = counts.most_common(1)[0]

correct = sum(
    1
    for label in true_labels
    if label == majority_intent
)

accuracy = correct / len(true_labels)

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "metric",
        "value"
    ])

    writer.writerow([
        "evaluation_examples",
        len(true_labels)
    ])

    writer.writerow([
        "majority_intent",
        majority_intent
    ])

    writer.writerow([
        "majority_count",
        majority_count
    ])

    writer.writerow([
        "accuracy",
        f"{accuracy:.4f}"
    ])


print("=" * 60)
print("TRIVIAL INTENT BASELINE")
print("=" * 60)
print(f"Evaluation examples: {len(true_labels)}")
print(f"Majority intent: {majority_intent}")
print(f"Majority count: {majority_count}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy * 100:.2f}%")
print(f"Output: {OUTPUT_FILE}")