import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


DATA_PATH = "processed/apple_customer_sample_500_labeled.csv"


def matched_rule(text):
    text = text.lower()

    # Transaction — keep highly specific transactional phrases
    if any(x in text for x in [
        "refund",
        "accidentally purchased",
        "return my",
        "warranty",
        "repair",
        "replacement",
        "replace my",
        "order",
        "ordered",
        "apple store appointment",
    ]):
        return "I9_transaction"

    # Battery
    if any(x in text for x in [
        "battery",
        "battery life",
        "battery drain",
        "battery draining",
        "charging",
        "charger",
        "charge",
    ]):
        return "I2_battery"

    # App
    if any(x in text for x in [
        "app store",
        "imovie",
        "spotify",
        "youtube",
        "facebook app",
    ]):
        return "I3_app"

    # Connectivity
    if any(x in text for x in [
        "wi-fi",
        "wifi",
        "bluetooth",
        "hotspot",
        "cellular",
    ]):
        return "I4_connectivity"

    # Communication
    if any(x in text for x in [
        "imessage",
        "text messages",
        "can't receive texts",
        "cannot receive texts",
        "not receiving texts",
    ]):
        return "I5_communication"

    # Account
    if any(x in text for x in [
        "apple id",
        "icloud",
        "account recovery",
        "password",
    ]):
        return "I7_account"

    # Service
    if any(x in text for x in [
        "apple music",
        "itunes",
        "ibooks",
        "podcast",
        "apple tv",
    ]):
        return "I8_service"

    # How-to
    if any(x in text for x in [
        "how do i",
        "how can i",
        "how to",
        "where can i",
        "how do you",
    ]):
        return "I10_howto"

    # Hardware — require more specific hardware phrases
    if any(x in text for x in [
        "home button",
        "touch screen",
        "touchscreen",
        "camera lens",
        "screen cracked",
        "display cracked",
    ]):
        return "I6_hardware"

    # Software — only very explicit software/update wording
    if any(x in text for x in [
        "ios update",
        "ios upgrade",
        "ios 11",
        "macos update",
        "high sierra",
        "software update",
    ]):
        return "I1_software"

    return None


df = pd.read_csv(DATA_PATH)

X = df["text"].fillna("")
y = df["intent"]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_tfidf = vectorizer.fit_transform(X)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

ml_predictions = cross_val_predict(
    model,
    X_tfidf,
    y,
    cv=cv
)

results = []

for text, true_label, ml_prediction in zip(
    X,
    y,
    ml_predictions
):
    rule = matched_rule(text)

    if rule is not None:
        results.append({
            "text": text,
            "true": true_label,
            "ml": ml_prediction,
            "rule": rule,
            "rule_correct": rule.split("_")[0] == true_label,
        })

results_df = pd.DataFrame(results)

print("=" * 70)
print("RULE PRECISION ANALYSIS")
print("=" * 70)

print(f"Messages matched by a rule: {len(results_df)}")

print("\nOverall rule accuracy:")
print(
    f"{results_df['rule_correct'].mean() * 100:.2f}%"
)

print("\nRule-by-rule precision:")

summary = (
    results_df
    .groupby("rule")
    .agg(
        matched=("rule_correct", "size"),
        correct=("rule_correct", "sum"),
    )
)

summary["precision"] = (
    summary["correct"] / summary["matched"] * 100
)

summary = summary.sort_values("precision")

print(summary.to_string())

print("\n" + "=" * 70)
print("LOW-PRECISION RULE EXAMPLES")
print("=" * 70)

bad = results_df[~results_df["rule_correct"]]

for _, row in bad.head(30).iterrows():
    print("\nRule:", row["rule"])
    print("True:", row["true"])
    print("ML:", row["ml"])
    print("Message:", row["text"])