import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


TRAIN_PATH = "processed/intent_train.csv"
TEST_PATH = "processed/intent_test.csv"


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["text"].fillna("")
y_train = train_df["intent"]

X_test = test_df["text"].fillna("")
y_test = test_df["intent"]


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(
    X_train_tfidf,
    y_train
)


probabilities = model.predict_proba(X_test_tfidf)

predictions = model.classes_[probabilities.argmax(axis=1)]
confidence = probabilities.max(axis=1)


results = pd.DataFrame({
    "true": y_test.values,
    "predicted": predictions,
    "confidence": confidence,
    "correct": predictions == y_test.values,
    "text": X_test.values
})


print("=" * 70)
print("INTENT CONFIDENCE ANALYSIS")
print("=" * 70)

print(f"Test examples: {len(results)}")

print("\nAverage confidence:")
print(
    f"Correct predictions: "
    f"{results.loc[results.correct, 'confidence'].mean():.3f}"
)

print(
    f"Incorrect predictions: "
    f"{results.loc[~results.correct, 'confidence'].mean():.3f}"
)


print("\nConfidence ranges:")

bins = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
labels = [
    "<0.40",
    "0.40-0.49",
    "0.50-0.59",
    "0.60-0.69",
    "0.70-0.79",
    "0.80+"
]

results["confidence_range"] = pd.cut(
    results["confidence"],
    bins=bins,
    labels=labels,
    right=False
)

summary = (
    results
    .groupby("confidence_range", observed=False)
    .agg(
        count=("correct", "size"),
        correct=("correct", "sum")
    )
)

summary["accuracy"] = (
    summary["correct"] / summary["count"] * 100
)

print(summary.to_string())


print("\n" + "=" * 70)
print("LOW CONFIDENCE EXAMPLES")
print("=" * 70)

low_conf = results.sort_values("confidence").head(20)

for i, (_, row) in enumerate(low_conf.iterrows(), start=1):

    print("\n" + "-" * 70)
    print(f"#{i}")

    print(f"True:       {row['true']}")
    print(f"Predicted:  {row['predicted']}")
    print(f"Confidence: {row['confidence']:.3f}")
    print(f"Correct:    {row['correct']}")
    print(f"Text:       {row['text']}")