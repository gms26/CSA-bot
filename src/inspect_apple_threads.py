import csv
import random

INPUT_PATH = "processed/apple_support_threads.csv"

NUM_THREADS = 10

print("=" * 70)
print("SAMPLE RECONSTRUCTED APPLESUPPORT THREADS")
print("=" * 70)

with open(INPUT_PATH, "r", encoding="utf-8", errors="replace", newline="") as file:
    reader = list(csv.DictReader(file))

print(f"\nTotal reconstructed threads: {len(reader):,}")

# Fixed seed makes our inspection reproducible
random.seed(42)

sample = random.sample(
    reader,
    min(NUM_THREADS, len(reader))
)

for i, row in enumerate(sample, start=1):

    print("\n" + "-" * 70)
    print(f"THREAD {i}")
    print("-" * 70)

    print(f"Thread ID   : {row['thread_id']}")
    print(f"Root Tweet  : {row['root_tweet_id']}")
    print(f"Customer ID : {row['customer_id']}")
    print(f"Messages    : {row['message_count']}")
    print(f"Start       : {row['start_time']}")
    print(f"End         : {row['end_time']}")

    print("\nConversation:")
    print(row["conversation"])

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)