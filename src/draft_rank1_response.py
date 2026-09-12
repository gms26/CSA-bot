import pandas as pd


INPUT_FILE = "processed/response_action_candidates.csv"
OUTPUT_FILE = "processed/rank1_grounded_drafts.csv"


def build_draft(query, response, actions):
    actions = str(actions).split("|")

    parts = []

    # Keep the acknowledgement style from the historical response.
    if "acknowledge" in actions:
        parts.append("We're here to help with this.")
    else:
        parts.append("We'd be happy to look into this with you.")

    # Preserve only actions actually present in Rank 1.
    if "ask_ios_version" in actions:
        parts.append(
            "Could you let us know which iOS version you're currently using?"
        )

    if "ask_device_model" in actions:
        parts.append(
            "Could you let us know which iPhone model you're using?"
        )

    if "ask_for_details" in actions:
        parts.append(
            "Could you tell us a little more about what happens when you try it?"
        )

    if "move_to_dm" in actions:
        parts.append(
            "Please send us a DM and we'll continue from there."
        )

    if "provide_resource" in actions:
        parts.append(
            "We can also point you to the relevant Apple Support resource."
        )

    if "troubleshooting" in actions:
        parts.append(
            "We'll work through the issue with you."
        )

    # If Rank 1 contained no explicit action.
    if len(parts) == 1:
        parts.append(
            "Let us know a few more details and we'll take a look."
        )

    return " ".join(parts)


def main():
    df = pd.read_csv(INPUT_FILE)

    # Only Rank 1 is used as grounding evidence.
    rank1 = (
        df[df["rank"] == 1]
        .copy()
        .sort_values("query")
    )

    drafts = []

    for _, row in rank1.iterrows():
        draft = build_draft(
            row["historical_customer_message"],
            row["historical_support_response"],
            row["response_actions"]
        )

        drafts.append({
            "customer_message":
                row["historical_customer_message"],
            "historical_response":
                row["historical_support_response"],
            "response_actions":
                row["response_actions"],
            "draft_response":
                draft
        })

    result = pd.DataFrame(drafts)

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nRank-1 grounded response drafts")
    print("--------------------------------")

    for _, row in result.iterrows():
        print("\n" + "=" * 80)
        print("CUSTOMER:")
        print(row["customer_message"])

        print("\nHISTORICAL RESPONSE:")
        print(row["historical_response"])

        print("\nACTIONS:")
        print(row["response_actions"])

        print("\nDRAFT:")
        print(row["draft_response"])

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()