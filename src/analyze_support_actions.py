import csv
import re
from collections import Counter

INPUT_PATH = "processed/apple_support_threads.csv"

# Simple phrase groups.
# These are for DATA EXPLORATION only.
# They are NOT our final intent labels.

ACTION_PATTERNS = {
    "dm_handoff": [
        r"\bdm\b",
        r"direct message",
        r"private message",
    ],

    "restart_device": [
        r"restart",
        r"reboot",
    ],

    "ios_version": [
        r"ios version",
        r"version of ios",
        r"what version of ios",
    ],

    "device_model": [
        r"which .*iphone",
        r"which .*ipad",
        r"which .*mac",
        r"device .*model",
        r"what .*model",
    ],

    "country_carrier": [
        r"country",
        r"carrier",
        r"region",
        r"located",
    ],

    "settings_check": [
        r"settings",
        r"check .* settings",
    ],

    "provide_article": [
        r"https://t\.co/",
        r"article",
        r"steps here",
    ],

    "more_information": [
        r"tell us more",
        r"more information",
        r"more info",
        r"more details",
        r"what .* happening",
        r"what exactly happens",
    ],

    "try_again": [
        r"try again",
        r"retry",
        r"then try",
    ],
}


def contains_pattern(text, patterns):
    text = text.lower()

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


total_threads = 0
support_messages = 0

action_counts = Counter()

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        total_threads += 1

        messages = row["conversation"].split("\n")

        for message in messages:

            if not message.startswith("APPLESUPPORT:"):
                continue

            support_messages += 1

            text = message[len("APPLESUPPORT:"):].strip()

            for action, patterns in ACTION_PATTERNS.items():

                if contains_pattern(text, patterns):
                    action_counts[action] += 1


print("=" * 70)
print("APPLE SUPPORT ACTION DISCOVERY")
print("=" * 70)

print(f"\nThreads examined          : {total_threads:,}")
print(f"AppleSupport messages    : {support_messages:,}")

print("\nObserved support actions:")

for action, count in action_counts.most_common():

    percentage = (
        count / support_messages * 100
        if support_messages
        else 0
    )

    print(
        f"{action:25} "
        f"{count:7,} "
        f"({percentage:5.1f}%)"
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)