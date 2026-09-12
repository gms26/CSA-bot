import csv
from collections import Counter

INPUT_FILE = "processed/apple_customer_sample_500.csv"

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

print("=" * 70)
print("APPLE SUPPORT INTENT TAXONOMY")
print("=" * 70)

print("\nProposed intents:")
for intent in INTENTS:
    print(" ", intent)

print("\nReading discovery sample...")

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Messages loaded: {len(rows)}")

print("\nThe taxonomy is currently a proposed labeling scheme.")
print("No labels are being assigned automatically.")
print("We will manually validate ambiguous examples before finalizing it.")

print("\n" + "=" * 70)
print("SAMPLE MESSAGES FOR MANUAL VALIDATION")
print("=" * 70)

# Show every 25th message so we can quickly test coverage.
for i, row in enumerate(rows, start=1):
    if i % 25 == 0:
        print(f"\n[{i}] {row['text']}")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)