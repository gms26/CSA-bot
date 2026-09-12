import pandas as pd
import re


INPUT_FILE = "processed/response_generation_candidates.csv"
OUTPUT_FILE = "processed/response_action_candidates.csv"


def extract_actions(response):
    text = str(response).lower()

    actions = []

    # DM / private support
    if (
        "dm" in text
        or "direct message" in text
        or "private message" in text
    ):
        actions.append("move_to_dm")

    # Device information
    if (
        "iphone" in text
        and (
            "model" in text
            or "which iphone" in text
            or "device" in text
        )
    ):
        actions.append("ask_device_model")

    # iOS version
    if (
        "ios version" in text
        or "version of ios" in text
        or "ios you're running" in text
        or "ios you are running" in text
    ):
        actions.append("ask_ios_version")

    # Error message / symptoms
    if (
        "error message" in text
        or "what happens" in text
        or "what happened" in text
        or "more details" in text
        or "more info" in text
    ):
        actions.append("ask_for_details")

    # Article / guide
    if (
        "article" in text
        or "guide" in text
        or "steps here" in text
        or "tips here" in text
    ):
        actions.append("provide_resource")

    # Troubleshooting
    if (
        "restart" in text
        or "reset" in text
        or "backup" in text
        or "update" in text
        or "check settings" in text
    ):
        actions.append("troubleshooting")

    # Acknowledgement / empathy
    if (
        "sorry" in text
        or "thanks for reaching out" in text
        or "understand" in text
        or "we're here to help" in text
        or "we'd be happy to help" in text
        or "we'd love to help" in text
    ):
        actions.append("acknowledge")

    if not actions:
        actions.append("general_help")

    return "|".join(actions)


def main():
    df = pd.read_csv(INPUT_FILE)

    df["response_actions"] = (
        df["historical_support_response"]
        .fillna("")
        .apply(extract_actions)
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nResponse action candidates")
    print("--------------------------")

    counts = (
        df["response_actions"]
        .str.split("|")
        .explode()
        .value_counts()
    )

    print(counts.to_string())

    print("\nExamples:\n")

    for action in counts.index:
        example = df[
            df["response_actions"].str.contains(
                action,
                regex=False
            )
        ].head(2)

        print("=" * 70)
        print("Action:", action)

        for _, row in example.iterrows():
            print("\nCustomer:")
            print(row["historical_customer_message"])

            print("\nResponse:")
            print(row["historical_support_response"])

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()