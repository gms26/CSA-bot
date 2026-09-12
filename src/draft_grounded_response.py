import pandas as pd


INPUT_FILE = "processed/response_action_candidates.csv"
OUTPUT_FILE = "processed/grounded_response_drafts.csv"


def build_draft(query, group):
    actions = []

    for value in group["response_actions"].dropna():
        actions.extend(value.split("|"))

    # Count actions across retrieved historical responses.
    action_counts = pd.Series(actions).value_counts()

    top_actions = action_counts.index.tolist()

    # Response style follows the strongest historical pattern.
    use_dm = "move_to_dm" in top_actions
    ask_ios = "ask_ios_version" in top_actions
    ask_device = "ask_device_model" in top_actions
    ask_details = "ask_for_details" in top_actions
    resource = "provide_resource" in top_actions

    parts = []

    # Acknowledge the issue.
    parts.append(
        "We're here to help with this."
    )

    # Ask for information supported by historical responses.
    questions = []

    if ask_device:
        questions.append("your iPhone model")

    if ask_ios:
        questions.append("your current iOS version")

    if ask_details:
        questions.append("what happens when you try it")

    if questions:
        if len(questions) == 1:
            parts.append(
                f"Could you send us {questions[0]} so we can look into it?"
            )
        else:
            joined = ", ".join(questions[:-1])
            joined += f" and {questions[-1]}"
            parts.append(
                f"Could you send us {joined} so we can look into it?"
            )

    # DM is only suggested when it appeared in retrieved history.
    if use_dm:
        parts.append(
            "Please send us a DM and we'll continue troubleshooting there."
        )

    # Resource is intentionally mentioned generically.
    # We do NOT invent a URL or article.
    elif resource:
        parts.append(
            "We can also point you to the relevant Apple support resource."
        )

    # If no specific information was historically requested,
    # ask the customer to describe the problem.
    if (
        not questions
        and not use_dm
        and not resource
    ):
        parts.append(
            "Let us know a few more details about the issue and we'll take a look."
        )

    return " ".join(parts)


def main():
    df = pd.read_csv(INPUT_FILE)

    drafts = []

    for query, group in df.groupby("query"):
        group = group.sort_values("rank")

        draft = build_draft(
            query,
            group
        )

        drafts.append({
            "customer_message": query,
            "draft_response": draft
        })

    result = pd.DataFrame(drafts)

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nGrounded response drafts")
    print("------------------------")

    for _, row in result.iterrows():
        print("\n" + "=" * 80)
        print("CUSTOMER:")
        print(row["customer_message"])
        print("\nDRAFT:")
        print(row["draft_response"])

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()