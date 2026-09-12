import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


INPUT_FILE = "processed/intent_train.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "intent"


class IntentClassifier:

    def __init__(self, input_file=INPUT_FILE):

        self.df = pd.read_csv(input_file)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )

        texts = self.df[TEXT_COLUMN].fillna("").astype(str)
        labels = self.df[LABEL_COLUMN].astype(str)

        self.X = self.vectorizer.fit_transform(texts)

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )

        self.model.fit(self.X, labels)

    def _high_confidence_override(self, text):

        text_lower = text.lower()

        # Clear connectivity/network problems.
        connectivity_terms = [
            "wi-fi",
            "wifi",
            "bluetooth",
            "hotspot",
        ]

        if any(term in text_lower for term in connectivity_terms):
            return "I4"

        # Clear transaction/return/repair cases.
        transaction_terms = [
            "return my iphone",
            "return my phone",
            "return this iphone",
            "return this phone",
            "replace my iphone",
            "replace my phone",
            "repair my iphone",
            "repair my phone",
            "warranty",
        ]

        if any(term in text_lower for term in transaction_terms):
            return "I9"

        return None

    def predict(self, text):

        override = self._high_confidence_override(text)

        if override is not None:
            text_vector = self.vectorizer.transform([text])
            probabilities = self.model.predict_proba(text_vector)[0]
            classes = self.model.classes_

            probability_map = dict(
                zip(classes, probabilities)
            )

            return {
                "intent": override,
                "probabilities": probability_map,
                "source": "high_confidence_rule",
            }

        text_vector = self.vectorizer.transform([text])

        prediction = self.model.predict(text_vector)[0]

        probabilities = self.model.predict_proba(text_vector)[0]

        classes = self.model.classes_

        probability_map = dict(
            zip(classes, probabilities)
        )

        return {
            "intent": prediction,
            "probabilities": probability_map,
            "source": "logistic_regression",
        }