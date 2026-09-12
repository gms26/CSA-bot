import csv
from collections import Counter


INPUT_FILE = "processed/apple_customer_sample_500_labeled.csv"


def contains_any(text, keywords):
    text = text.lower()
    return any(keyword in text for keyword in keywords)


def matched_rule(text):
    text_lower = text.lower()

    if contains_any(text_lower, [
        "refund", "accidentally purchased", "purchase", "purchased",
        "order", "ordered", "warranty", "repair", "replacement",
        "replace my", "return my", "apple store appointment",
        "customer service", "call back"
    ]):
        return "I9_transaction"

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
            return "I2_battery"

    if contains_any(text_lower, [
        "app store", "appstore", "apps not", "apps won't",
        "apps wont", "app won't", "app wont", "app crashing",
        "app crashes", "imovie", "spotify", "youtube", "facebook app"
    ]):
        return "I3_app"

    if contains_any(text_lower, [
        "wi-fi", "wifi", "bluetooth", "hotspot", "cellular",
        "4g", "5g", "network", "internet connection",
        "can't connect", "cannot connect"
    ]):
        return "I4_connectivity"

    if contains_any(text_lower, [
        "text", "texts", "imessage", "i message", "sms",
        "messages", "calls", "calling", "phone calls",
        "can't receive calls", "cannot receive calls"
    ]):
        return "I5_communication"

    if contains_any(text_lower, [
        "apple id", "icloud", "password", "account", "sign in",
        "signin", "login", "locked out", "security"
    ]):
        return "I7_account"

    if contains_any(text_lower, [
        "apple music", "itunes", "ibooks", "podcast", "podcasts",
        "apple tv", "tv app", "music", "voicemail"
    ]):
        return "I8_service"

    if contains_any(text_lower, [
        "how do i", "how can i", "where do i", "how to",
        "what is", "can i change", "settings", "enable",
        "disable", "turn on", "turn off"
    ]):
        return "I10_howto"

    if contains_any(text_lower, [
        "screen", "display", "camera", "home button", "button",
        "keyboard", "speaker", "microphone", "water damage",
        "dropped", "broken", "physical", "won't turn on",
        "won’t turn on"
    ]):
        return "I6_hardware"

    if contains_any(text_lower, [
        "ios", "ios update", "ios upgrade", "macos", "mac os",
        "high sierra", "software update", "after update",
        "latest update", "system update", "freezing", "frozen",
        "crashing", "slow", "lagging"
    ]):
        return "I1_software"

    return None


# ---------------------------------------------------------
# Train the same ML model used previously
# ---------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)


texts = [row["text"] for row in rows]
labels = [row["intent"] for row in rows]


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


# ---------------------------------------------------------
# Find regressions
# ---------------------------------------------------------

regressions = []

for row, ml_prediction in zip(rows, ml_predictions):

    rule = matched_rule(row["text"])

    if rule is None:
        continue

    true_intent = row["intent"]

    ml_correct = ml_prediction == true_intent
    rule_correct = rule.split("_")[0] == true_intent

    if ml_correct and not rule_correct:

        regressions.append({
            "rule": rule,
            "true": true_intent,
            "ml": ml_prediction,
            "rule_prediction": rule.split("_")[0],
            "text": row["text"]
        })


print("=" * 70)
print("RULE REGRESSION ANALYSIS")
print("=" * 70)

print(f"Total rule regressions: {len(regressions)}")


counts = Counter(item["rule"] for item in regressions)

print("\nRegressions by rule:")

for rule, count in counts.most_common():
    print(f"{rule}: {count}")


print("\n" + "=" * 70)
print("REGRESSION EXAMPLES")
print("=" * 70)


for i, item in enumerate(regressions[:40], start=1):

    print("\n" + "-" * 70)
    print(f"REGRESSION #{i}")
    print(f"Rule: {item['rule']}")
    print(f"True: {item['true']}")
    print(f"ML: {item['ml']}")
    print(f"Rule prediction: {item['rule_prediction']}")
    print(f"Message: {item['text']}")