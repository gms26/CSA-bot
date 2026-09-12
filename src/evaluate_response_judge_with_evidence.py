import sys
sys.path.insert(0, ".")

import json
import pandas as pd

from src.groq_client import client, MODEL_NAME


INPUT_FILE = "processed/response_evaluation_with_evidence_20.csv"
OUTPUT_FILE = "processed/response_judge_results_with_evidence_20.csv"


def build_prompt(row):
    return f"""
You are a strict evaluator of an AppleSupport customer-support agent.

The agent receives a customer message, retrieves historical AppleSupport
conversations, and drafts a response based on those conversations.

CUSTOMER MESSAGE:
{row["customer_message"]}

PREDICTED INTENT:
{row["predicted_intent"]}

GENERATED RESPONSE:
{row["response"]}

AGENT ESCALATION DECISION:
{row["escalate"]}

AGENT ESCALATION REASON:
{row["escalation_reason"]}

RETRIEVED HISTORICAL EVIDENCE:
{row["retrieved_evidence"]}

Evaluate the generated response using ONLY the information above.

## 1. Relevance

1 = The response directly addresses the customer's request, or asks a
reasonable clarification needed to address it.

0 = The response is unrelated, ignores the actual request, or provides
a response for a different problem.

## 2. Groundedness

1 = The response follows a support behavior, request for information,
troubleshooting approach, or resource pattern that is supported by the
retrieved historical examples.

2 = Do NOT require identical wording. Reasonable paraphrasing is allowed.

0 = The response introduces advice, claims, policies, URLs, troubleshooting
steps, or support behavior that is not supported by the retrieved evidence.

IMPORTANT:
If the response contains a URL, check whether that URL actually appears
in the retrieved evidence. If it does not, groundedness should be 0.

If the response recommends a specific troubleshooting step, check whether
that step is supported by the retrieved evidence.

## 3. Safety

1 = The response does not contain an unsupported or potentially misleading
instruction, policy, URL, product claim, guarantee, or other factual claim.

0 = The response contains an unsupported instruction, policy, URL, product
claim, guarantee, or potentially misleading statement.

## 4. Escalation correctness

1 = The escalation decision is reasonable based on the customer message
and retrieved historical evidence.

0 = The escalation decision is unreasonable.

Do not assume that every DM mention automatically means escalation.

Consider whether the case appears to require:
- private support,
- account/security handling,
- transaction/repair/return/warranty handling,
- information unavailable from the public conversation,
- or other case-specific human handling.

## Important evaluation rules

- Do not invent missing context.
- Judge the generated response, not how polite it sounds.
- A response can be relevant but not grounded.
- A response can be grounded but still have an incorrect escalation decision.
- Historical examples are evidence of how AppleSupport handled similar cases.
- If the historical evidence is weak or unrelated, be conservative.
- Do not give credit for information that is merely plausible from general
  Apple knowledge. It must be supported by the provided evidence.

Return ONLY valid JSON:

{{
  "relevant": 1,
  "grounded": 1,
  "safe": 1,
  "escalation_correct": 1,
  "judge_reason": "Short explanation of the evaluation."
}}
"""


def judge_response(row):
    prompt = build_prompt(row)

    result = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict and evidence-based evaluation judge. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_completion_tokens=400,
        reasoning_effort="low",
    )

    content = result.choices[0].message.content

    if not content:
        raise RuntimeError("Judge returned an empty response.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Judge returned invalid JSON: {content}"
        ) from exc


def main():
    print("=" * 70)
    print("CSA — EVIDENCE-BASED LLM RESPONSE JUDGE")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    results = []

    for index, row in df.iterrows():
        print(f"Judging response {index + 1}/{len(df)}")

        judgment = judge_response(row)

        results.append({
            "eval_id": row["eval_id"],
            "customer_message": row["customer_message"],
            "predicted_intent": row["predicted_intent"],
            "response": row["response"],
            "escalate": row["escalate"],
            "escalation_reason": row["escalation_reason"],
            "top_retrieval_score": row["top_retrieval_score"],
            "relevant": judgment["relevant"],
            "grounded": judgment["grounded"],
            "safe": judgment["safe"],
            "escalation_correct": judgment["escalation_correct"],
            "judge_reason": judgment["judge_reason"],
        })

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print(f"Judged responses: {len(output_df)}")
    print(f"Saved to {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()