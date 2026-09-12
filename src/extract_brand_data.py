import csv
from collections import Counter

INPUT_PATH = "raw/twcs/twcs.csv"
OUTPUT_PATH = "processed/brand_candidates.csv"

BRANDS = {
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
    "SouthwestAir",
    "VirginTrains",
    "Ask_Spectrum",
    "XboxSupport",
    "sprintcare",
    "hulu_support",
    "sainsburys",
    "GWRHelp",
    "AskPlayStation",
    "ChipotleTweets",
    "VerizonSupport",
    "UPSHelp",
    "ATVIAssist",
    "O2",
    "Safaricom_Care",
    "idea_cares",
    "AskTarget",
    "AirAsiaSupport",
    "BofA_Help",
    "SW_Help",
}

# How many messages to collect for each brand.
TARGET_PER_BRAND = 5_000

counts = Counter()

columns_to_keep = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

print("=" * 70)
print("EXTRACTING BRAND DATA")
print("=" * 70)

print("\nReading raw dataset...\n")

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
    errors="replace",
    newline="",
) as infile:

    reader = csv.DictReader(infile)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as outfile:

        writer = csv.DictWriter(
            outfile,
            fieldnames=columns_to_keep,
        )

        writer.writeheader()

        processed = 0

        for row in reader:

            processed += 1

            author = row["author_id"]

            # Keep messages written by our known support accounts.
            if author in BRANDS:

                # Stop collecting once we have enough for this brand.
                if counts[author] < TARGET_PER_BRAND:

                    writer.writerow({
                        column: row[column]
                        for column in columns_to_keep
                    })

                    counts[author] += 1

            # Print progress every 500k rows.
            if processed % 500_000 == 0:
                print(
                    f"Processed {processed:,} rows..."
                )

            # Stop once every brand has enough messages.
            if len(counts) == len(BRANDS):
                if all(
                    counts[brand] >= TARGET_PER_BRAND
                    for brand in BRANDS
                ):
                    break


print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(f"\nRows scanned: {processed:,}")

print("\nMessages collected per brand:")

for brand, count in counts.most_common():
    print(f"{brand:<25} {count:,}")

print(f"\nOutput:")
print(OUTPUT_PATH)