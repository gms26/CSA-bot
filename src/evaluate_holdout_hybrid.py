import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


TRAIN_PATH = "processed/intent_train.csv"
TEST_PATH = "processed/intent_test.csv"


def high_confidence_rule(text):
    """
    Return an intent only when the wording is highly specific.
    Otherwise return None and allow ML to decide.
    """

    text = text.lower()

    # I5 — Communication
    if any(x in text for x in [
        "imessage",
        "text messages",
        "not receiving texts",
        "can't receive texts",
        "cannot receive texts",
    ]):
        return "I5"

    # I7 — Account / iCloud / security
    if any(x in text for x in [
        "apple id",
        "icloud",
        "account recovery",
    ]):
        return "I7"

    # I9 — Transaction / repair / warranty
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

    # I4 — Connectivity
    if any(x in text for x in [
        "wi-fi",
        "wifi",
        "bluetooth",
        "hotspot",
    ]):
        return "I4"

    # I2 — Battery
    if any(x in text for x in [
        "battery life",
        "battery drain",
        "battery draining",
        "won't charge",
        "will not charge",
        "charging",
    ]):
        return "I2"

    # I3 — Specific app/store
    if any(x in text for x in [
        "app store",
        "imovie",
        "spotify",
    ]):
        return "I3"

    # I8 — Specific Apple service/media
    if any(x in text for x in [
        "apple music",
        "ibooks",
        "apple tv",
        "podcasts",
    ]):
        return "I8"

    # I6 — Very specific physical hardware
    if any(x in text for x in [
        "screen cracked",
        "display cracked",
        "home button",
        "camera lens",
    ]):
        return "I6"

    # I10 — Explicit how-to question
    if any(x in text for x in [
        "how do i",
        "how can i",
        "how to",
    ]):
        return "I10"

    # I1 is intentionally NOT overridden by generic
    # words such as "ios", "update", "crash", etc.
    #
    # Those words occur inside many other intents.

    return None


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["text"].fillna("")
y_train = train_df["intent"]

X_test = test_df["text"].fillna("")
y_test = test_df["intent"]


# ---------------------------------------------------------
# Train ML model using training set only
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Apply high-confidence rules
# ---------------------------------------------------------

hybrid_predictions = []
rules_used = []

for text, ml_prediction in zip(X_test, ml_predictions):

    rule_prediction = high_confidence_rule(text)

    if rule_prediction is not None:
        hybrid_predictions.append(rule_prediction)
        rules_used.append(rule_prediction)
    else:
        hybrid_predictions.append(ml_prediction)
        rules_used.append("ML")


# ---------------------------------------------------------
# Evaluate ML baseline
# ---------------------------------------------------------

ml_accuracy = accuracy_score(y_test, ml_predictions)

ml_macro_f1 = f1_score(
    y_test,
    ml_predictions,
    average="macro",
    zero_division=0
)

ml_weighted_f1 = f1_score(
    y_test,
    ml_predictions,
    average="weighted",
    zero_division=0
)


# ---------------------------------------------------------
# Evaluate hybrid
# ---------------------------------------------------------

hybrid_accuracy = accuracy_score(
    y_test,
    hybrid_predictions
)

hybrid_macro_f1 = f1_score(
    y_test,
    hybrid_predictions,
    average="macro",
    zero_division=0
)

hybrid_weighted_f1 = f1_score(
    y_test,
    hybrid_predictions,
    average="weighted",
    zero_division=0
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("=" * 70)
print("HOLDOUT HYBRID INTENT EVALUATION")
print("=" * 70)

print(f"Training examples: {len(train_df)}")
print(f"Test examples:     {len(test_df)}")

print("\n" + "-" * 70)
print("ML BASELINE")
print("-" * 70)

print(f"Accuracy:    {ml_accuracy * 100:.2f}%")
print(f"Macro F1:    {ml_macro_f1:.2f}")
print(f"Weighted F1: {ml_weighted_f1:.2f}")

print("\n" + "-" * 70)
print("HYBRID")
print("-" * 70)

print(f"Accuracy:    {hybrid_accuracy * 100:.2f}%")
print(f"Macro F1:    {hybrid_macro_f1:.2f}")
print(f"Weighted F1: {hybrid_weighted_f1:.2f}")

print("\n" + "-" * 70)
print("RULE USAGE")
print("-" * 70)

rule_count = sum(
    1 for rule in rules_used
    if rule != "ML"
)

print(f"Rules used: {rule_count}/{len(test_df)}")
print(f"ML used:    {len(test_df) - rule_count}/{len(test_df)}")

print("\nRule distribution:")

rule_series = pd.Series(rules_used)

print(
    rule_series.value_counts().to_string()
)

print("\n" + "-" * 70)
print("HYBRID CLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_test,
        hybrid_predictions,
        zero_division=0
    )
)