import csv

INPUT_FILE = "processed/apple_support_response_pairs.csv"

START = 1
END = 30

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    rows = list(reader)

for i, row in enumerate(rows[START - 1:END], start=START):
    print("=" * 70)
    print(f"PAIR #{i}")
    print(f"Thread ID: {row['thread_id']}")
    print(f"Customer: {row['customer_message']}")
    print(f"AppleSupport: {row['support_response']}")