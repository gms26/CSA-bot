import csv
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "processed/apple_customer_sample_500_labeled.csv"


def contains_any(text, keywords):
    text = text.lower()
    return any(keyword in text for keyword in keywords)


def rule_based_intent(text):
    text_lower = text.lower()

    # ---------------------------------------------------------
    # I9: Purchase / Order / Repair / Warranty
    # High-priority transactional signals
    # ---------------------------------------------------------

    i9_keywords = [
        "refund",
        "refund me",
        "accidentally purchased",
        "accidental purchase",
        "purchase",
        "purchased",
        "order",
        "ordered",
        "warranty",
        "repair",
        "replacement",
        "replace my",
        "return my",
        "apple store appointment",
        "customer service",
        "call back",
    ]

    if contains_any(text_lower, i9_keywords):
        return "I9"


    # ---------------------------------------------------------
    # I2: Battery / Charging / Power
    # ---------------------------------------------------------

    i2_keywords = [
        "battery",
        "battery life",
        "draining battery",
        "battery draining",
        "won't charge",
        "won’t charge",
        "not charging",
        "charge above",
        "charging",
        "charger",
    ]

    if contains_any(text_lower, i2_keywords):

        # If the message clearly describes a broad OS/update
        # problem with multiple symptoms, don't force I2.
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


    # ---------------------------------------------------------
    # I3: App / App Store / App Update
    # ---------------------------------------------------------

    i3_keywords = [
        "app store",
        "appstore",
        "apps not",
        "apps won't",
        "apps wont",
        "app won't",
        "app wont",
        "app crashing",
        "app crashes",
        "imovie",
        "spotify",
        "youtube",
        "facebook app",
    ]

    if contains_any(text_lower, i3_keywords):
        return "I3"


    # ---------------------------------------------------------
    # I4: Connectivity / Network
    # ---------------------------------------------------------

    i4_keywords = [
        "wi-fi",
        "wifi",
        "bluetooth",
        "hotspot",
        "cellular",
        "4g",
        "5g",
        "network",
        "internet connection",
        "can't connect",
        "cannot connect",
    ]

    if contains_any(text_lower, i4_keywords):
        return "I4"


    # ---------------------------------------------------------
    # I5: Messages / Communication
    # ---------------------------------------------------------

    i5_keywords = [
        "text",
        "texts",
        "imessage",
        "i message",
        "sms",
        "messages",
        "calls",
        "calling",
        "phone calls",
        "can't receive calls",
        "cannot receive calls",
    ]

    if contains_any(text_lower, i5_keywords):
        return "I5"


    # ---------------------------------------------------------
    # I7: Account / iCloud / Security
    # ---------------------------------------------------------

    i7_keywords = [
        "apple id",
        "icloud",
        "password",
        "account",
        "sign in",
        "signin",
        "login",
        "locked out",
        "security",
    ]

    if contains_any(text_lower, i7_keywords):
        return "I7"


    # ---------------------------------------------------------
    # I8: Apple Services / Media
    # ---------------------------------------------------------

    i8_keywords = [
        "apple music",
        "itunes",
        "ibooks",
        "podcast",
        "podcasts",
        "apple tv",
        "tv app",
        "music",
        "voicemail",
    ]

    if contains_any(text_lower, i8_keywords):
        return "I8"


    # ---------------------------------------------------------
    # I10: How-to / Settings / Information
    # ---------------------------------------------------------

    i10_keywords = [
        "how do i",
        "how can i",
        "where do i",
        "how to",
        "what is",
        "can i change",
        "settings",
        "enable",
        "disable",
        "turn on",
        "turn off",
    ]

    if contains_any(text_lower, i10_keywords):
        return "I10"


    # ---------------------------------------------------------
    # I6: Device / Hardware
    # ---------------------------------------------------------

    i6_keywords = [
        "screen",
        "display",
        "camera",
        "home button",
        "button",
        "keyboard",
        "speaker",
        "microphone",
        "water damage",
        "dropped",
        "broken",
        "physical",
        "won't turn on",
        "won’t turn on",
    ]

    if contains_any(text_lower, i6_keywords):
        return "I6"


    # ---------------------------------------------------------
    # I1: Software / iOS / macOS
    # ---------------------------------------------------------

    i1_keywords = [
        "ios",
        "ios update",
        "ios upgrade",
        "macos",
        "mac os",
        "high sierra",
        "software update",
        "after update",
        "latest update",
        "system update",
        "freezing",
        "frozen",
        "crashing",
        "slow",
        "lagging",
    ]

    if contains_any(text_lower, i1_keywords):
        return "I1"


    # No strong rule
    return None


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading labelled examples...")

rows = []

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)


texts = [row["text"] for row in rows]
labels = [row["intent"] for row in rows]


# ---------------------------------------------------------
# Train/evaluate ML baseline with cross-validation
# ---------------------------------------------------------

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


print("Running cross-validation...")

ml_predictions = cross_val_predict(
    pipeline,
    texts,
    labels,
    cv=cv
)


# ---------------------------------------------------------
# Hybrid prediction
# ---------------------------------------------------------

hybrid_predictions = []

rule_used = 0

for text, ml_prediction in zip(texts, ml_predictions):

    rule_prediction = rule_based_intent(text)

    if rule_prediction is not None:
        hybrid_predictions.append(rule_prediction)
        rule_used += 1
    else:
        hybrid_predictions.append(ml_prediction)


accuracy = accuracy_score(labels, hybrid_predictions)


print("=" * 60)
print("HYBRID INTENT CLASSIFIER")
print("=" * 60)

print(f"Examples: {len(labels)}")
print(f"Rules used: {rule_used}")
print(f"Rules used (%): {rule_used / len(labels) * 100:.2f}%")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy * 100:.2f}%")

print("\nClassification report:")

print(
    classification_report(
        labels,
        hybrid_predictions,
        zero_division=0
    )
)