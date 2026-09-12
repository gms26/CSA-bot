import pandas as pd
import sys

# Allow Python to import our existing label dictionary
sys.path.insert(0, "src")

from analyze_intent_distribution import labels


INPUT_FILE = "processed/apple_customer_sample_500_labeled.csv"


# Read the 500-message CSV
df = pd.read_csv(INPUT_FILE)


# Safety check
if len(df) != 500:
    raise ValueError(f"Expected 500 rows, found {len(df)}")


# Put the reviewed intent into each row
df["intent"] = df["sample_id"].map(labels)


# Check that every message received a label
if df["intent"].isna().any():
    missing = df.loc[df["intent"].isna(), "sample_id"].tolist()
    raise ValueError(f"Missing labels for: {missing}")


# Default confidence
df["label_confidence"] = "high"


# Cases that we specifically reviewed as borderline
review_cases = {
    164, 211, 218, 291, 313,
    330, 350, 358, 377, 391,
    399, 407, 420, 425, 451,
    452, 473, 474, 496, 497
}

df.loc[
    df["sample_id"].isin(review_cases),
    "label_confidence"
] = "review"


# Explanation for each intent
reason_map = {
    "I1": "System-level software, iOS, macOS, or update-related problem.",
    "I2": "Battery, charging, or power is the primary issue.",
    "I3": "Specific app, App Store, installation, or app update problem.",
    "I4": "Wi-Fi, cellular, Bluetooth, hotspot, or other connectivity problem.",
    "I5": "Messages, iMessage, email, or communication functionality problem.",
    "I6": "Device hardware or device-level physical functionality problem.",
    "I7": "Apple account, iCloud, password, or security-related problem.",
    "I8": "Apple service or media-related problem.",
    "I9": "Purchase, order, refund, repair, replacement, or warranty problem.",
    "I10": "How-to, configuration, settings, or information request.",
    "I11": "Insufficient context, generic response, or unclear problem."
}

df["label_reason"] = df["intent"].map(reason_map)


# Save the completed CSV
df.to_csv(INPUT_FILE, index=False)


print("=" * 60)
print("INTENT LABELS APPLIED")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Labeled: {df['intent'].notna().sum()}")
print(f"Output: {INPUT_FILE}")
print()
print("Distribution:")
print(df["intent"].value_counts().sort_index())