import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = "processed/apple_support_response_pairs_clean.csv"
EMBEDDINGS_FILE = "processed/apple_support_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticRetriever:
    def __init__(self, input_file=INPUT_FILE):
        self.df = pd.read_csv(input_file)

        self.model = SentenceTransformer(MODEL_NAME)

        if os.path.exists(EMBEDDINGS_FILE):
            print("Loading cached embeddings...")
            self.embeddings = __import__("numpy").load(
                EMBEDDINGS_FILE
            )
        else:
            print("Creating embeddings...")
            customer_texts = (
                self.df["customer_message"]
                .fillna("")
                .astype(str)
                .tolist()
            )

            self.embeddings = self.model.encode(
                customer_texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

            __import__("numpy").save(
                EMBEDDINGS_FILE,
                self.embeddings,
            )

            print(
                f"Saved embeddings to {EMBEDDINGS_FILE}"
            )

        if len(self.embeddings) != len(self.df):
            raise RuntimeError(
                "Embedding count does not match response-pair count."
            )

    def retrieve(self, query, top_k=5, exclude_thread_id=None):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        scores = cosine_similarity(
            query_embedding,
            self.embeddings,
        )[0]

        candidate_indices = scores.argsort()[::-1]

        results = []

        for index in candidate_indices:
            row = self.df.iloc[index]

            if (
                exclude_thread_id is not None
                and str(row["thread_id"]) == str(exclude_thread_id)
            ):
                continue

            results.append(
                {
                    "thread_id": row["thread_id"],
                    "customer_message": row["customer_message"],
                    "support_response": row["support_response"],
                    "similarity": float(scores[index]),
                }
            )

            if len(results) >= top_k:
                break

        return results