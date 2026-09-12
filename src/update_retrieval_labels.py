import pandas as pd

INPUT_FILE = "processed/semantic_retrieval_labeling_100.csv"

labels = {
    86: [1, 1, 1, 1, 1],
    87: [0, 0, 0, 0, 0],
    88: [1, 1, 1, 1, 1],
    89: [1, 1, 1, 1, 1],
    90: [1, 1, 1, 1, 1],
    91: [1, 1, 1, 1, 1],
    92: [0, 1, 1, 1, 1],
    93: [1, 1, 1, 1, 1],
    94: [1, 1, 1, 0, 1],
    95: [1, 1, 1, 1, 1],
    96: [0, 0, 0, 0, 0],
    97: [1, 1, 1, 1, 1],
    98: [1, 1, 1, 1, 1],
    99: [1, 1, 1, 1, 0],
    100: [0, 1, 1, 1, 1],
}

df = pd.read_csv(INPUT_FILE)

for eval_id, relevance_values in labels.items():
    rows = df["eval_id"] == eval_id

    if rows.sum() != 5:
        raise ValueError(
            f"Q{eval_id} has {rows.sum()} rows instead of 5."
        )

    indices = df.loc[rows].sort_values("rank").index

    for index, relevance in zip(indices, relevance_values):
        df.loc[index, "relevance"] = relevance

df.to_csv(INPUT_FILE, index=False)

print("Successfully updated Q56-Q70.")