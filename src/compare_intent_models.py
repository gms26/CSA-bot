import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


INPUT_FILE = "processed/apple_customer_sample_500_labeled.csv"


def contains_any(text, keywords):
    text = text.lower()
    return any(keyword in text for keyword in keywords)


def rule_based_intent(text):
    text_lower = text.lower()

    if contains_any(text_lower, [
        "refund", "accidentally purchased", "purchase", "purchased",
        "order", "ordered", "warranty", "repair", "replacement",
        "replace my", "return my", "apple store appointment",
        "customer service", "call back"
    ]):
        return "I9"

    if contains_any(text_lower, [
        "battery", "battery life", "draining battery",
        "battery draining", "won't charge", "won’t charge",
        "not charging", "charge above", "charging", "charger"
    ]):
        broad_update = (
            ("ios" in text_lower or "update" in text_lower)
            and (
                "app" in text_lower
                or "apps" in text_lower
                or "crash" in text_lower
                or "freez" in text_lower
                or "lag" in text_lower
                or "cellular" in text_lower
            )
        )

        if not broad_update:
            return "I2"

    if contains_any(text_lower, [
        "app store", "appstore", "apps not", "apps won't",
        "apps wont", "app won't", "app wont", "app crashing",
        "app crashes", "imovie", "spotify", "youtube", "facebook app"
    ]):
        return "I3"

    if contains_any(text_lower, [
        "wi-fi", "wifi", "bluetooth", "hotspot", "cellular",
        "4g", "5g", "network", "internet connection",
        "can't connect", "cannot connect"
    ]):
        return "I4"

    if contains_any(text_lower, [
        "text", "texts", "imessage", "i message", "sms",
        "messages", "calls", "calling", "phone calls",
        "can't receive calls", "cannot receive calls"
    ]):
        return "I5"

    if contains_any(text_lower, [
        "apple id", "icloud", "password", "account", "sign in",
        "signin", "login", "locked out", "security"
    ]):
        return "I7"

    if contains_any(text_lower, [
        "apple music", "itunes", "ibooks", "podcast", "podcasts",
        "apple tv", "tv app", "music", "voicemail"
    ]):
        return "I8"

    if contains_any(text_lower, [
        "how do i", "how can i", "where do i", "how to",
        "what is", "can i change", "settings", "enable",
        "disable", "turn on", "turn off"
    ]):
        return "I10"

    if contains_any(text_lower, [
        "screen", "display", "camera", "home button", "button",
        "keyboard", "speaker", "microphone", "water damage",
        "dropped", "broken", "physical", "won't turn on",
        "won’t turn on"
    ]):
        return "I6"

    if contains_any(text_lower, [
        "ios", "ios update", "ios upgrade", "macos", "mac os",
        "high sierra", "software update", "after update",
        "latest update", "system update", "freezing", "frozen",
        "crashing", "slow", "lagging"
    ]):
        return "I1"

    return None


# Load data

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

texts = [row["text"] for row in rows]
labels = [row["intent"] for row in rows]


# ML model

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


ml_predictions = cross_val_predict(
    pipeline,
    texts,
    labels,
    cv=cv
)


# Hybrid predictions

hybrid_predictions = []

for text, ml_prediction in zip(texts, ml_predictions):

    rule_prediction = rule_based_intent(text)

    if rule_prediction is not None:
        hybrid_predictions.append(rule_prediction)
    else:
        hybrid_predictions.append(ml_prediction)


# Compare

improved = []
worsened = []

for row, ml_prediction, hybrid_prediction in zip(
    rows,
    ml_predictions,
    hybrid_predictions
):

    true_intent = row["intent"]

    ml_correct = ml_prediction == true_intent
    hybrid_correct = hybrid_prediction == true_intent

    if not ml_correct and hybrid_correct:
        improved.append({
            "id": row["tweet_id"],
            "true": true_intent,
            "ml": ml_prediction,
            "hybrid": hybrid_prediction,
            "text": row["text"]
        })

    elif ml_correct and not hybrid_correct:
        worsened.append({
            "id": row["tweet_id"],
            "true": true_intent,
            "ml": ml_prediction,
            "hybrid": hybrid_prediction,
            "text": row["text"]
        })


print("=" * 70)
print("HYBRID MODEL COMPARISON")
print("=" * 70)

print(f"ML correct: {sum(p == y for p, y in zip(ml_predictions, labels))}")
print(
    f"Hybrid correct: "
    f"{sum(p == y for p, y in zip(hybrid_predictions, labels))}"
)

print(f"Cases improved by rules: {len(improved)}")
print(f"Cases worsened by rules: {len(worsened)}")


print("\n" + "=" * 70)
print("CASES IMPROVED BY RULES")
print("=" * 70)

for i, item in enumerate(improved[:25], start=1):

    print("\n" + "-" * 70)
    print(f"IMPROVED #{i}")
    print(f"ID: {item['id']}")
    print(f"True: {item['true']}")
    print(f"ML: {item['ml']}")
    print(f"Hybrid: {item['hybrid']}")
    print(f"Message: {item['text']}")


print("\n" + "=" * 70)
print("CASES WORSENED BY RULES")
print("=" * 70)

for i, item in enumerate(worsened[:25], start=1):

    print("\n" + "-" * 70)
    print(f"WORSENED #{i}")
    print(f"ID: {item['id']}")
    print(f"True: {item['true']}")
    print(f"ML: {item['ml']}")
    print(f"Hybrid: {item['hybrid']}")
    print(f"Message: {item['text']}")