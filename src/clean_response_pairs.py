import csv
import re

INPUT_FILE = "processed/apple_support_response_pairs.csv"
OUTPUT_FILE = "processed/apple_support_response_pairs_clean.csv"


def clean_text(text):
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove Twitter usernames
    text = re.sub(r"@\w+", "", text)

    # Replace common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("&quot;", '"')

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


count = 0

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as outfile:

    reader = csv.DictReader(infile)

    fieldnames = [
        "thread_id",
        "customer_id",
        "customer_message",
        "support_response",
    ]

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        customer_message = clean_text(row["customer_message"])
        support_response = clean_text(row["support_response"])

        if not customer_message or not support_response:
            continue

        writer.writerow({
            "thread_id": row["thread_id"],
            "customer_id": row["customer_id"],
            "customer_message": customer_message,
            "support_response": support_response,
        })

        count += 1


print("=" * 60)
print("CLEAN APPLE SUPPORT RESPONSE PAIRS")
print("=" * 60)
print(f"Clean pairs created: {count}")
print(f"Output: {OUTPUT_FILE}")