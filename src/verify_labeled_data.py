import pandas as pd

FILE = "processed/apple_customer_sample_500_labeled.csv"

df = pd.read_csv(FILE)

print("=" * 60)
print("LABELED DATA VERIFICATION")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print()
print("Missing intents:", df["intent"].isna().sum())
print("Missing confidence:", df["label_confidence"].isna().sum())
print("Missing reasons:", df["label_reason"].isna().sum())

print()
print("Intent distribution:")
print(df["intent"].value_counts().sort_index())

print()
print("First 5 labeled examples:")
print(df[["sample_id", "text", "intent", "label_confidence"]].head().to_string(index=False))