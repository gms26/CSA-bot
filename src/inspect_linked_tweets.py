import csv

INPUT_PATH = "raw/twcs/twcs.csv"

TARGET_IDS = {
    "698",
    "704",
    "707",
    "709",
    "702",
    "717",
    "719",
    "721",
    "714",
}

found = {}

print("=" * 70)
print("INSPECTING LINKED APPLESUPPORT TWEETS")
print("=" * 70)

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
    errors="replace",
    newline="",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        tweet_id = row["tweet_id"]

        if tweet_id in TARGET_IDS:
            found[tweet_id] = row

        # Stop early once all target tweets are found.
        if len(found) == len(TARGET_IDS):
            break


for tweet_id in sorted(
    found,
    key=lambda x: int(x)
):

    row = found[tweet_id]

    print("\n" + "-" * 70)

    print(f"Tweet ID: {row['tweet_id']}")
    print(f"Author: {row['author_id']}")
    print(f"Inbound: {row['inbound']}")

    print(f"\nMessage:")
    print(row["text"])

    print(f"\nResponse Tweet ID:")
    print(row["response_tweet_id"])

    print(f"\nIn Response To:")
    print(row["in_response_to_tweet_id"])


print("\n" + "=" * 70)
print(f"Found {len(found)} of {len(TARGET_IDS)} target tweets.")
print("=" * 70)