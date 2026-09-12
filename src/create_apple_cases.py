import csv

INPUT_PATH = "processed/apple_support_threads.csv"
OUTPUT_PATH = "processed/apple_support_cases.csv"

total = 0
kept = 0

with open(INPUT_PATH, "r", encoding="utf-8", errors="replace", newline="") as file:
    reader = csv.DictReader(file)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as output:

        fieldnames = reader.fieldnames + ["pattern"]

        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in reader:

            total += 1

            message_count = int(row["message_count"])

            # Need at least:
            # Customer → AppleSupport → Customer
            if message_count < 3:
                continue

            messages = row["conversation"].split("\n")

            speakers = []

            for message in messages:
                if message.startswith("CUSTOMER:"):
                    speakers.append("C")
                elif message.startswith("APPLESUPPORT:"):
                    speakers.append("A")

            # Require a clean alternating conversation
            alternating = all(
                speakers[i] != speakers[i + 1]
                for i in range(len(speakers) - 1)
            )

            if not alternating:
                continue

            # Root must be customer
            if not speakers or speakers[0] != "C":
                continue

            # Must contain AppleSupport
            if "A" not in speakers:
                continue

            # Must contain customer follow-up after AppleSupport
            first_support = speakers.index("A")

            if "C" not in speakers[first_support + 1:]:
                continue

            row["pattern"] = " → ".join(speakers)

            writer.writerow(row)

            kept += 1

print("=" * 70)
print("APPLE SUPPORT CASE CREATION")
print("=" * 70)

print(f"\nTotal threads examined : {total:,}")
print(f"Clean cases kept      : {kept:,}")
print(f"Filtered out          : {total - kept:,}")

print("\nOutput:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)