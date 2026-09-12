import pandas as pd
import re


INPUT_FILE = "processed/apple_support_response_pairs_clean.csv"


def classify_response(text):
    text = str(text).lower()

    patterns = {
        "dm_or_private_support": [
            r"\bdm\b",
            r"direct message",
            r"private message",
            r"message us"
        ],
        "request_more_information": [
            r"can you.*tell",
            r"could you.*tell",
            r"let us know",
            r"provide.*details",
            r"more information",
            r"more info"
        ],
        "troubleshooting": [
            r"restart",
            r"reset",
            r"try again",
            r"check.*settings",
            r"check.*update",
            r"update.*device"
        ],
        "article_or_link": [
            r"help\.apple\.com",
            r"support\.apple\.com",
            r"article",
            r"guide"
        ],
        "acknowledgement": [
            r"sorry",
            r"understand",
            r"thanks",
            r"thank you",
            r"apolog"
        ]
    }

    labels = []

    for label, regexes in patterns.items():
        if any(re.search(pattern, text) for pattern in regexes):
            labels.append(label)

    if not labels:
        return "other"

    return "|".join(labels)


def main():
    df = pd.read_csv(INPUT_FILE)

    df["response_pattern"] = (
        df["support_response"]
        .fillna("")
        .apply(classify_response)
    )

    counts = (
        df["response_pattern"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["response_pattern", "count"]

    counts["percentage"] = (
        counts["count"] / len(df) * 100
    ).round(2)

    print("\nHistorical AppleSupport response patterns")
    print("-----------------------------------------")
    print("Response pairs:", len(df))
    print()

    print(counts.to_string(index=False))

    print("\nExamples:\n")

    for pattern in counts["response_pattern"].head(10):
        example = df[
            df["response_pattern"] == pattern
        ].head(2)

        print("=" * 80)
        print("Pattern:", pattern)

        for _, row in example.iterrows():
            print("\nCustomer:")
            print(row["customer_message"])
            print("\nAppleSupport:")
            print(row["support_response"])

    counts.to_csv(
        "processed/response_pattern_counts.csv",
        index=False
    )

    print(
        "\nSaved: processed/response_pattern_counts.csv"
    )


if __name__ == "__main__":
    main()