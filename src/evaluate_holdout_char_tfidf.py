import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


TRAIN_PATH = "processed/intent_train.csv"
TEST_PATH = "processed/intent_test.csv"


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["text"].fillna("")
y_train = train_df["intent"]

X_test = test_df["text"].fillna("")
y_test = test_df["intent"]


# ---------------------------------------------------------
# Word-level TF-IDF
# ---------------------------------------------------------

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)


# ---------------------------------------------------------
# Character-level TF-IDF
# ---------------------------------------------------------

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)


# ---------------------------------------------------------
# Combine features
# ---------------------------------------------------------

X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])


# ---------------------------------------------------------
# Train classifier
# ---------------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(
    X_train_combined,
    y_train
)

predictions = model.predict(X_test_combined)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

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
print("HOLDOUT WORD + CHARACTER TF-IDF")
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