import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


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


weights = [
    ("balanced", "balanced"),
    ("uniform", None),
    ("custom", {
        "I1": 1.0,
        "I10": 2.0,
        "I11": 1.0,
        "I2": 1.5,
        "I3": 2.0,
        "I4": 2.0,
        "I5": 2.0,
        "I6": 2.0,
        "I7": 2.0,
        "I8": 2.0,
        "I9": 2.0,
    }),
]


print("=" * 70)
print("CLASS WEIGHT EXPERIMENT")
print("=" * 70)

for name, class_weight in weights:

    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n" + "-" * 70)
    print(name)

    print(f"Accuracy:    {accuracy * 100:.2f}%")
    print(f"Macro F1:    {macro_f1:.2f}")
    print(f"Weighted F1: {weighted_f1:.2f}")