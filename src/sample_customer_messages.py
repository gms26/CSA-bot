import csv
import random

INPUT_PATH = "processed/apple_support_raw.csv"
OUTPUT_PATH = "processed/apple_customer_sample_500.csv"

SAMPLE_SIZE = 500

customers = []

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row["inbound"].lower() == "true":

            customers.append({
                "tweet_id": row["tweet_id"],
                "author_id": row["author_id"],
                "created_at": row["created_at"],
                "text": row["text"],
                "response_tweet_id": row["response_tweet_id"],
                "in_response_to_tweet_id": row[
                    "in_response_to_tweet_id"
                ]
            })


print("=" * 70)
print("CUSTOMER MESSAGE SAMPLING")
print("=" * 70)

print(f"\nCustomer messages available: {len(customers):,}")

random.seed(42)

sample = random.sample(
    customers,
    min(SAMPLE_SIZE, len(customers))
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    fieldnames = [
        "sample_id",
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for index, row in enumerate(sample, start=1):

        row["sample_id"] = index

        writer.writerow({
            "sample_id": row["sample_id"],
            "tweet_id": row["tweet_id"],
            "author_id": row["author_id"],
            "created_at": row["created_at"],
            "text": row["text"],
            "response_tweet_id": row["response_tweet_id"],
            "in_response_to_tweet_id": row[
                "in_response_to_tweet_id"
            ]
        })


print(f"Random sample created: {len(sample):,}")
print(f"\nOutput:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("SAMPLING COMPLETE")
print("=" * 70)