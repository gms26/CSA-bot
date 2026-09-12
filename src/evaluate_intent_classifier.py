import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "processed/apple_customer_sample_500_labeled.csv"
OUTPUT_FILE = "processed/intent_classifier_predictions.csv"


print("Loading labelled examples...")

rows = []

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

texts = [row["text"] for row in rows]
labels = [row["intent"] for row in rows]


print(f"Examples: {len(texts)}")
print(f"Intents: {len(set(labels))}")


pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("Running 5-fold stratified cross-validation...")

predictions = cross_val_predict(
    pipeline,
    texts,
    labels,
    cv=cv
)


accuracy = accuracy_score(labels, predictions)


print("=" * 60)
print("TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        labels,
        predictions,
        zero_division=0
    )
)


print("\nSaving predictions...")

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:

    fieldnames = [
        "tweet_id",
        "text",
        "true_intent",
        "predicted_intent",
        "correct"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for row, prediction in zip(rows, predictions):

        writer.writerow({
            "tweet_id": row["tweet_id"],
            "text": row["text"],
            "true_intent": row["intent"],
            "predicted_intent": prediction,
            "correct": prediction == row["intent"]
        })


print(f"Output: {OUTPUT_FILE}")