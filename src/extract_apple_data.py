import csv

INPUT_PATH = "raw/twcs/twcs.csv"
OUTPUT_PATH = "processed/apple_support_raw.csv"

TARGET_BRAND = "AppleSupport"

# Store tweet IDs that are directly connected to AppleSupport.
apple_tweet_ids = set()

print("=" * 70)
print("STEP 1: FIND APPLESUPPORT TWEETS")
print("=" * 70)

# ------------------------------------------------------------
# PASS 1
# Find all tweets written by AppleSupport.
# ------------------------------------------------------------

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
    errors="replace",
    newline="",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row["author_id"] == TARGET_BRAND:
            apple_tweet_ids.add(row["tweet_id"])

print(f"\nAppleSupport tweets found: {len(apple_tweet_ids):,}")


# ------------------------------------------------------------
# PASS 2
# Extract:
#
# 1. AppleSupport tweets
# 2. Tweets directly connected to AppleSupport tweets
# ------------------------------------------------------------

columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

connected_ids = set(apple_tweet_ids)

print("\nFinding connected customer tweets...")

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
    errors="replace",
    newline="",
) as file:

    reader = csv.DictReader(file)

    rows_to_write = []

    for row in reader:

        tweet_id = row["tweet_id"]

        # Keep AppleSupport tweets.
        if tweet_id in apple_tweet_ids:
            rows_to_write.append(row)
            continue

        # Check whether this tweet responds to AppleSupport.
        parent_id = row["in_response_to_tweet_id"]

        if parent_id in apple_tweet_ids:
            rows_to_write.append(row)
            connected_ids.add(tweet_id)
            continue

        # Check whether AppleSupport responds to this tweet.
        response_ids = row["response_tweet_id"]

        if response_ids:

            response_list = [
                x.strip()
                for x in response_ids.split(",")
            ]

            if any(
                response_id in apple_tweet_ids
                for response_id in response_list
            ):
                rows_to_write.append(row)
                connected_ids.add(tweet_id)


print(
    f"Connected tweets collected: {len(rows_to_write):,}"
)


# ------------------------------------------------------------
# Write result
# ------------------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=columns,
    )

    writer.writeheader()

    writer.writerows(rows_to_write)


print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(f"\nOutput:")
print(OUTPUT_PATH)