import pandas as pd

FILE = "processed/apple_customer_intent_review.csv"

df = pd.read_csv(FILE)

review = df[df["label_confidence"] == "review"]

print("=" * 70)
print("BORDERLINE INTENT REVIEW")
print("=" * 70)
print(f"Total cases for review: {len(review)}")
print()

for _, row in review.iterrows():
    print(f"[{int(row['sample_id'])}]")
    print(f"Intent: {row['intent']}")
    print(f"Text: {row['text']}")
    print("-" * 70)