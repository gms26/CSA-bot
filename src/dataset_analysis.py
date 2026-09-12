import pandas as pd

DATA_PATH = "raw/sample.csv"
print("=" * 60)
print("SAMPLE DATASET INSPECTION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nInbound distribution:")
print(df["inbound"].value_counts())

print("\nTop authors:")
print(df["author_id"].value_counts().head(30))