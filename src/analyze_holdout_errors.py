import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


TRAIN_PATH = "processed/intent_train.csv"
TEST_PATH = "processed/intent_test.csv"


def high_confidence_rule(text):
    text = text.lower()

    if any(x in text for x in [
        "imessage",
        "text messages",
        "not receiving texts",
        "can't receive texts",
        "cannot receive texts",
    ]):
        return "I5"

    if any(x in text for x in [
        "apple id",
        "icloud",
        "account recovery",
    ]):
        return "I7"

    if any(x in text for x in [
        "refund",
        "accidentally purchased",
        "warranty",
        "repair",
        "replacement",
        "replace my",
        "return my",
        "apple store appointment",
    ]):
        return "I9"

    if any(x in text for x in [
        "wi-fi",
        "wifi",
        "bluetooth",
        "hotspot",
    ]):
        return "I4"

    if any(x in text for x in [
        "battery life",
        "battery drain",
        "battery draining",
        "won't charge",
        "will not charge",
        "charging",
    ]):
        return "I2"

    if any(x in text for x in [
        "app store",
        "imovie",
        "spotify",
    ]):
        return "I3"

    if any(x in text for x in [
        "apple music",
        "ibooks",
        "apple tv",
        "podcasts",
    ]):
        return "I8"

    if any(x in text for x in [
        "screen cracked",
        "display cracked",
        "home button",
        "camera lens",
    ]):
        return "I6"

    if any(x in text for x in [
        "how do i",
        "how can i",
        "how to",
    ]):
        return "I10"

    return None


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

model.fit(X_train_tfidf, y_train)

ml_predictions = model.predict(X_test_tfidf)

rows = []

for text, true_label, ml_prediction in zip(
    X_test,
    y_test,
    ml_predictions
):
    rule_prediction = high_confidence_rule(text)

    if rule_prediction is not None:
        final_prediction = rule_prediction
        source = "RULE"
    else:
        final_prediction = ml_prediction
        source = "ML"

    if final_prediction != true_label:
        rows.append({
            "true": true_label,
            "ml": ml_prediction,
            "rule": rule_prediction or "NONE",
            "final": final_prediction,
            "source": source,
            "text": text
        })


errors = pd.DataFrame(rows)

print("=" * 70)
print("HOLDOUT ERROR ANALYSIS")
print("=" * 70)

print(f"Test examples: {len(test_df)}")
print(f"Errors:        {len(errors)}")
print(f"Correct:       {len(test_df) - len(errors)}")

print("\n" + "-" * 70)
print("CONFUSION PAIRS")
print("-" * 70)

confusions = (
    errors
    .groupby(["true", "final"])
    .size()
    .sort_values(ascending=False)
)

print(confusions.to_string())

print("\n" + "-" * 70)
print("ERRORS BY SOURCE")
print("-" * 70)

print(errors["source"].value_counts().to_string())

print("\n" + "=" * 70)
print("ALL HOLDOUT ERRORS")
print("=" * 70)

for i, (_, row) in enumerate(errors.iterrows(), start=1):

    print("\n" + "-" * 70)
    print(f"ERROR #{i}")
    print("-" * 70)

    print("True: ", row["true"])
    print("ML:   ", row["ml"])
    print("Rule: ", row["rule"])
    print("Final: ", row["final"])
    print("Source:", row["source"])
    print("Text: ", row["text"])