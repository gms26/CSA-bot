import csv
from collections import Counter

INPUT_PATH = "processed/apple_support_threads.csv"

total_multiturn = 0
alternating = 0
has_customer_followup = 0

patterns = Counter()

with open(INPUT_PATH, "r", encoding="utf-8", errors="replace", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:

        message_count = int(row["message_count"])

        if message_count <= 2:
            continue

        total_multiturn += 1

        messages = row["conversation"].split("\n")

        speakers = []

        for message in messages:
            if message.startswith("CUSTOMER:"):
                speakers.append("C")
            elif message.startswith("APPLESUPPORT:"):
                speakers.append("A")

        pattern = " → ".join(speakers)
        patterns[pattern] += 1

        # Check whether the conversation alternates
        is_alternating = all(
            speakers[i] != speakers[i + 1]
            for i in range(len(speakers) - 1)
        )

        if is_alternating:
            alternating += 1

        # Customer appears after the first AppleSupport response
        if "A" in speakers:
            first_support = speakers.index("A")

            if "C" in speakers[first_support + 1:]:
                has_customer_followup += 1


print("=" * 70)
print("MULTI-TURN APPLESUPPORT THREAD ANALYSIS")
print("=" * 70)

print(f"\nMulti-turn threads          : {total_multiturn:,}")
print(f"Alternating conversations  : {alternating:,}")
print(f"Customer follow-up exists  : {has_customer_followup:,}")

print("\nConversation patterns:")

for pattern, count in patterns.most_common(20):
    print(f"{count:6,}  {pattern}")

print("\n" + "=" * 70)