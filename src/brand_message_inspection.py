import pandas as pd

DATA_PATH = "processed/brand_candidates.csv"

df = pd.read_csv(DATA_PATH)

brands = [
    "AmazonHelp",
    "AppleSupport",
]

for brand in brands:

    print("\n")
    print("=" * 80)
    print(f"BRAND: {brand}")
    print("=" * 80)

    brand_df = df[df["author_id"] == brand]

    print(f"\nTotal messages available: {len(brand_df)}")

    print("\nSample brand responses:\n")

    samples = brand_df.sample(
        n=min(30, len(brand_df)),
        random_state=42,
    )

    for i, text in enumerate(samples["text"], start=1):

        print(f"{i}. {text}")
        print("-" * 80)