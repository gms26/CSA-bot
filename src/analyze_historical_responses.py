import csv
from collections import Counter

FILE = "processed/apple_support_threads.csv"

total_threads = 0
total_support_messages = 0

response_patterns = Counter()

with open(FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        total_threads += 1

        conversation = row["conversation"]

        messages = conversation.split("\n")

        support_messages = []

        for message in messages:
            if message.startswith("APPLESUPPORT:"):
                support_messages.append(message)

        total_support_messages += len(support_messages)

        for message in support_messages:
            text = message.lower()

            if "dm" in text or "direct message" in text:
                response_patterns["DM / private support"] += 1

            elif "try" in text or "restart" in text:
                response_patterns["Troubleshooting instruction"] += 1

            elif "update" in text:
                response_patterns["Update-related guidance"] += 1

            elif "check" in text or "see if" in text:
                response_patterns["Diagnostic / check"] += 1

            elif "sorry" in text or "apolog" in text:
                response_patterns["Apology / acknowledgement"] += 1

            else:
                response_patterns["Other support response"] += 1


print("=" * 60)
print("HISTORICAL APPLESUPPORT RESPONSE ANALYSIS")
print("=" * 60)

print(f"Threads: {total_threads}")
print(f"AppleSupport messages: {total_support_messages}")

print()
print("Observed response patterns:")

for pattern, count in response_patterns.most_common():
    percentage = (
        count / total_support_messages * 100
        if total_support_messages
        else 0
    )

    print(f"{pattern}: {count} ({percentage:.1f}%)")