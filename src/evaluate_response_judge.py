import sys
sys.path.insert(0, ".")

import json
import pandas as pd

from src.groq_client import client, MODEL_NAME


INPUT_FILE = "processed/response_evaluation_20.csv"
OUTPUT_FILE = "processed/response_judge_results_20.csv"


def build_prompt(row):
    return f"""
You are evaluating an AppleSupport customer-support agent.

Customer message:
{row["customer_message"]}

Predicted intent:
{row["predicted_intent"]}

Generated response:
{row["response"]}

Agent escalation decision:
{row["escalate"]}

Agent escalation reason:
{row["escalation_reason"]}

Retrieved historical evidence:
The agent retrieved historical AppleSupport examples before generating
the response. Use the following information as the available evidence:

Top retrieval similarity:
{row["top_retrieval_score"]}

Evaluate four dimensions.

1. Relevance
1 = The response directly addresses the customer's request or asks a
reasonable clarification.
0 = It is unrelated, misleading, or fails to address the request.

2. Groundedness
1 = The response follows a support behavior or guidance pattern supported
by the historical AppleSupport evidence.
0 = It introduces behavior, advice, or claims not supported by the evidence.

3. Safety
1 = It does not invent unsupported troubleshooting steps, policies,
URLs, product facts, or guarantees.
0 = It contains unsupported instructions, policies, URLs, product claims,
or other potentially misleading information.

4. Escalation correctness
1 = The escalation decision is reasonable for this customer case.
0 = The escalation decision is unreasonable.

Important:
- Do not assume every DM mention requires escalation.
- Do not reward politeness alone.
- Do not invent missing context.
- Be conservative when evidence is insufficient.

Return ONLY valid JSON:

{{
  "relevant": 1,
  "grounded": 1,
  "safe": 1,
  "escalation_correct": 1,
  "judge_reason": "Short explanation."
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
                    "You are a strict evaluation judge. "
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
    print("CSA — LLM RESPONSE JUDGE")
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