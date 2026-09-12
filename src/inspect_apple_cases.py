import csv
import random

INPUT_PATH = "processed/apple_support_cases.csv"

NUM_CASES = 10

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as file:
    cases = list(csv.DictReader(file))

print("=" * 70)
print("SAMPLE CLEAN APPLESUPPORT CASES")
print("=" * 70)

print(f"\nTotal clean cases: {len(cases):,}")

random.seed(42)

sample = random.sample(
    cases,
    min(NUM_CASES, len(cases))
)

for i, case in enumerate(sample, start=1):

    print("\n" + "-" * 70)
    print(f"CASE {i}")
    print("-" * 70)

    print(f"Thread ID : {case['thread_id']}")
    print(f"Pattern   : {case['pattern']}")
    print(f"Messages  : {case['message_count']}")

    print("\nConversation:")
    print(case["conversation"])

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)