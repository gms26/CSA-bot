import pandas as pd

FILE = "processed/semantic_retrieval_labeling_100.csv"

df = pd.read_csv(
    FILE,
    dtype={
        "relevance": "string",
        "label_reason": "string"
    }
)

mask = df["eval_id"] == 1

df.loc[mask, "relevance"] = "0"
df.loc[mask, "label_reason"] = (
    "Context-dependent query; retrieved cases cannot be confirmed "
    "as addressing the same problem."
)

df.to_csv(FILE, index=False)

print("Evaluation Query 1 labeled successfully.")
print(
    df[df["eval_id"] == 1][
        ["eval_id", "rank", "relevance", "label_reason"]
    ].to_string(index=False)
)