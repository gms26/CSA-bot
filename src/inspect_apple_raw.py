import csv
from collections import Counter

INPUT_PATH = "processed/apple_support_raw.csv"

total = 0
inbound_count = 0
outbound_count = 0

authors = Counter()
missing_parent = 0
missing_response = 0

print("=" * 70)
print("APPLE SUPPORT RAW DATA INSPECTION")
print("=" * 70)

with open(INPUT_PATH, "r", encoding="utf-8", errors="replace", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        total += 1

        author = row["author_id"]
        authors[author] += 1

        if row["inbound"].lower() == "true":
            inbound_count += 1
        else:
            outbound_count += 1

        if not row["in_response_to_tweet_id"]:
            missing_parent += 1

        if not row["response_tweet_id"]:
            missing_response += 1

print(f"\nTotal extracted tweets : {total:,}")
print(f"Customer tweets        : {inbound_count:,}")
print(f"AppleSupport tweets    : {outbound_count:,}")

print(f"\nMissing parent links   : {missing_parent:,}")
print(f"Missing response links: {missing_response:,}")

print("\nTop authors:")
for author, count in authors.most_common(15):
    print(f"{author:25} {count:,}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)