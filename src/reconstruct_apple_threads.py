import csv
from collections import defaultdict

INPUT_PATH = "processed/apple_support_raw.csv"
OUTPUT_PATH = "processed/apple_support_threads.csv"

print("=" * 70)
print("RECONSTRUCT APPLESUPPORT ROOTED THREADS")
print("=" * 70)

# ------------------------------------------------------------
# STEP 1: Load extracted tweets
# ------------------------------------------------------------

tweets = {}

children = defaultdict(list)

with open(INPUT_PATH, "r", encoding="utf-8", errors="replace", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        tweet_id = row["tweet_id"]

        tweets[tweet_id] = row

        parent_id = row["in_response_to_tweet_id"]

        if parent_id:
            children[parent_id].append(tweet_id)

print(f"\nTweets loaded: {len(tweets):,}")

# ------------------------------------------------------------
# STEP 2: Find customer root tweets
# ------------------------------------------------------------

root_ids = []

for tweet_id, row in tweets.items():

    if row["inbound"].lower() != "true":
        continue

    # Root = customer tweet with no parent
    if not row["in_response_to_tweet_id"]:
        root_ids.append(tweet_id)

print(f"Customer root candidates: {len(root_ids):,}")

# ------------------------------------------------------------
# STEP 3: Follow response paths
# ------------------------------------------------------------

threads = []

for root_id in root_ids:

    path = []
    current_id = root_id
    visited = set()

    while current_id and current_id not in visited:

        visited.add(current_id)

        if current_id not in tweets:
            break

        row = tweets[current_id]
        path.append(row)

        next_ids = children.get(current_id, [])

        if not next_ids:
            break

        # Sort possible responses chronologically
        next_ids = sorted(
            next_ids,
            key=lambda x: tweets[x]["created_at"]
        )

        # Follow the earliest response path
        current_id = next_ids[0]

    # --------------------------------------------------------
    # Only keep conversations containing AppleSupport
    # --------------------------------------------------------

    if not any(
        row["author_id"] == "AppleSupport"
        for row in path
    ):
        continue

    # Need at least customer + AppleSupport
    if len(path) < 2:
        continue

    threads.append(path)

print(f"Rooted threads reconstructed: {len(threads):,}")

# ------------------------------------------------------------
# STEP 4: Write threads
# ------------------------------------------------------------

columns = [
    "thread_id",
    "root_tweet_id",
    "customer_id",
    "message_count",
    "start_time",
    "end_time",
    "conversation"
]

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for index, path in enumerate(threads, start=1):

        root = path[0]

        conversation_parts = []

        for row in path:

            speaker = (
                "CUSTOMER"
                if row["inbound"].lower() == "true"
                else "APPLESUPPORT"
            )

            conversation_parts.append(
                f"{speaker}: {row['text']}"
            )

        writer.writerow({
            "thread_id": f"apple_{index:06d}",
            "root_tweet_id": root["tweet_id"],
            "customer_id": root["author_id"],
            "message_count": len(path),
            "start_time": path[0]["created_at"],
            "end_time": path[-1]["created_at"],
            "conversation": "\n".join(conversation_parts)
        })

print("\n" + "=" * 70)
print("RECONSTRUCTION COMPLETE")
print("=" * 70)

print(f"\nOutput:")
print(OUTPUT_PATH)