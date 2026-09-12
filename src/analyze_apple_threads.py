import csv
from collections import Counter

INPUT_PATH = "processed/apple_support_threads.csv"

message_counts = Counter()

total = 0
single_response = 0
multi_turn = 0

with open(INPUT_PATH, "r", encoding="utf-8", errors="replace", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        total += 1

        count = int(row["message_count"])
        message_counts[count] += 1

        if count == 2:
            single_response += 1
        elif count > 2:
            multi_turn += 1

print("=" * 70)
print("APPLE SUPPORT THREAD ANALYSIS")
print("=" * 70)

print(f"\nTotal threads       : {total:,}")
print(f"2-message threads   : {single_response:,}")
print(f"Multi-turn threads  : {multi_turn:,}")

print("\nMessage count distribution:")

for count in sorted(message_counts):
    print(f"{count:2} messages : {message_counts[count]:,}")

print("\n" + "=" * 70)