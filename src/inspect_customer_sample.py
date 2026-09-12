import csv

INPUT_PATH = "processed/apple_customer_sample_500.csv"

START = 400
END = 500

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as file:

    reader = csv.DictReader(file)
    rows = list(reader)

print("=" * 70)
print("CUSTOMER INTENT DISCOVERY SAMPLE")
print("=" * 70)

print(f"\nShowing messages {START} to {END}")
print(f"Total available: {len(rows)}")

for row in rows[START - 1:END]:

    print(
        f"\n[{row['sample_id']}] "
        f"{row['text']}"
    )

print("\n" + "=" * 70)