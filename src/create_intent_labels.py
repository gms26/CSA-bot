import csv

INPUT_FILE = "processed/apple_customer_sample_500.csv"
OUTPUT_FILE = "processed/apple_customer_sample_500_labeled.csv"

INTENTS = [
    "I1 - Software / iOS / macOS Issue",
    "I2 - Battery / Charging / Power",
    "I3 - App / App Store Issue",
    "I4 - Connectivity / Network",
    "I5 - Messages / Communication",
    "I6 - Device / Hardware Issue",
    "I7 - Apple Account / iCloud / Security",
    "I8 - Media / Apple Services",
    "I9 - Purchase / Order / Repair / Warranty",
    "I10 - How-to / Settings / Information",
    "I11 - Other / Unclear",
]

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

fieldnames = list(rows[0].keys()) + ["intent", "label_confidence", "label_reason"]

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        row["intent"] = ""
        row["label_confidence"] = ""
        row["label_reason"] = ""
        writer.writerow(row)

print("=" * 70)
print("INTENT LABELING FILE CREATED")
print("=" * 70)
print(f"Input messages : {len(rows)}")
print(f"Output file    : {OUTPUT_FILE}")
print()
print("Intent columns have been added but are intentionally empty.")
print("We will label the messages manually.")
print()
print("Available intents:")
for intent in INTENTS:
    print(" ", intent)