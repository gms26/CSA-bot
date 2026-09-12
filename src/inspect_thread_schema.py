import csv

FILE = "processed/apple_support_threads.csv"

with open(FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    print("=" * 60)
    print("APPLE SUPPORT THREAD SCHEMA")
    print("=" * 60)

    print("Columns:")
    for column in reader.fieldnames:
        print(f" - {column}")

    print()
    print("First 3 threads:")
    print("-" * 60)

    for i, row in enumerate(reader):
        print(row)

        if i == 2:
            break