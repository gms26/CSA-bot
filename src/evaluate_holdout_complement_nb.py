import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, classification_report


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


model = ComplementNB()

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


print("=" * 70)
print("HOLDOUT COMPLEMENT NAIVE BAYES")
print("=" * 70)

print(f"Training examples: {len(train_df)}")
print(f"Test examples:     {len(test_df)}")

print(f"\nAccuracy:    {accuracy * 100:.2f}%")
print(f"Macro F1:    {macro_f1:.2f}")
print(f"Weighted F1: {weighted_f1:.2f}")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)
