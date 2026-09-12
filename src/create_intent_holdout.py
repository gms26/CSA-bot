import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "processed/apple_customer_sample_500_labeled.csv"

TRAIN_OUTPUT = "processed/intent_train.csv"
TEST_OUTPUT = "processed/intent_test.csv"

df = pd.read_csv(INPUT)

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["intent"],
    random_state=42
)

train_df.to_csv(TRAIN_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)

print("=" * 60)
print("INTENT HOLDOUT CREATED")
print("=" * 60)

print(f"Total: {len(df)}")
print(f"Train: {len(train_df)}")
print(f"Test:  {len(test_df)}")

print("\nTraining distribution:")
print(train_df["intent"].value_counts().sort_index())

print("\nTest distribution:")
print(test_df["intent"].value_counts().sort_index())