import csv

INPUT_FILE = "processed/apple_support_threads.csv"
OUTPUT_FILE = "processed/apple_support_response_pairs.csv"

pair_count = 0

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

        conversation = row["conversation"]

        messages = conversation.split("\n")

        previous_customer = None

        for message in messages:

            if message.startswith("CUSTOMER:"):
                previous_customer = message.replace(
                    "CUSTOMER:", "", 1
                ).strip()

            elif message.startswith("APPLESUPPORT:"):

                support_response = message.replace(
                    "APPLESUPPORT:", "", 1
                ).strip()

                if previous_customer:

                    writer.writerow({
                        "thread_id": row["thread_id"],
                        "customer_id": row["customer_id"],
                        "customer_message": previous_customer,
                        "support_response": support_response,
                    })

                    pair_count += 1

                previous_customer = None


print("=" * 60)
print("APPLE SUPPORT RESPONSE PAIRS")
print("=" * 60)

print(f"Response pairs created: {pair_count}")
print(f"Output: {OUTPUT_FILE}")