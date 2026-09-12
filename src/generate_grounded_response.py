import json
import re

from src.groq_client import client, MODEL_NAME

def _clean_text(text):
    """Clean common formatting artifacts without changing the meaning."""
    if not text:
        return ""

    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€”": "—",
        "â€“": "–",
        "â€¦": "...",
        "â€¯": " ",
        "ï¿½": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Fix a few common concatenation artifacts produced by generation.
    fixes = {
        "ifyou": "if you",
        "wecan": "we can",
        "youcan": "you can",
        "helpwith": "help with",
        "throughthe": "through the",
        "tothe": "to the",
        "forthe": "for the",
        "withthe": "with the",
        "onthe": "on the",
        "inthe": "in the",
        "fromthe": "from the",
        "ofthe": "of the",
        "humanhandling": "human handling",
        "directmessage": "direct message",
    }

    for old, new in fixes.items():
        text = text.replace(old, new)

    # Remove accidental leading/trailing quotes around the complete response.
    text = text.strip()

    if len(text) >= 2:
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _normalise_for_comparison(text):
    """
    Create a lightweight representation for comparing the generated response
    with historical evidence.

    This is intentionally simple. We do not require exact wording because
    legitimate paraphrasing is allowed.
    """
    text = text.lower()

    # Remove URLs and punctuation.
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    words = text.split()

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "get",
        "have",
        "help",
        "hi",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "so",
        "that",
        "the",
        "this",
        "to",
        "us",
        "we",
        "what",
        "with",
        "you",
        "your",
    }

    return {
        word
        for word in words
        if len(word) >= 3 and word not in stop_words
    }


def _evidence_similarity(generated, historical):
    """
    Calculate a conservative lexical overlap score.

    This is NOT an accuracy metric. It is only a safety guard used to detect
    responses that are almost completely unrelated to the supplied evidence.
    """
    generated_words = _normalise_for_comparison(generated)
    historical_words = _normalise_for_comparison(historical)

    if not generated_words or not historical_words:
        return 0.0

    intersection = generated_words.intersection(historical_words)

    return len(intersection) / len(generated_words)


def _contains_unsupported_instruction(response, evidence):
    """
    Detect common cases where the model invents a concrete instruction that
    was not present in the supplied historical response.

    This is deliberately conservative. It is a guardrail, not a semantic
    verifier.
    """
    response_lower = response.lower()
    evidence_lower = evidence.lower()

    instruction_patterns = [
        r"\btap\s+(?:the\s+)?",
        r"\bclick\s+(?:on\s+)?",
        r"\bgo\s+to\s+",
        r"\bopen\s+(?:the\s+)?",
        r"\bselect\s+(?:the\s+)?",
        r"\bchoose\s+(?:the\s+)?",
        r"\bnavigate\s+to\s+",
        r"\bturn\s+(?:on|off)\s+",
        r"\benable\s+",
        r"\bdisable\s+",
        r"\bhold\s+",
        r"\bpress\s+",
        r"\bswipe\s+",
        r"\bupdate\s+",
        r"\brestart\s+",
        r"\breset\s+",
        r"\bdelete\s+",
        r"\breinstall\s+",
    ]

    for pattern in instruction_patterns:
        if re.search(pattern, response_lower):
            # If the same instruction language appears in the historical
            # evidence, it is supported.
            if not re.search(pattern, evidence_lower):
                return True

    return False


def _safe_fallback(historical_responses):
    """
    If generation cannot be safely grounded, return the strongest historical
    response rather than inventing advice.

    The historical response is already brand-authored evidence.
    """
    if not historical_responses:
        return (
            "Could you share a little more information about the issue "
            "so we can help?"
        )

    first = historical_responses[0]

    if isinstance(first, dict):
        response = first.get("support_response", "")
    else:
        response = str(first)

    response = _clean_text(response)

    if not response:
        return (
            "Could you share a little more information about the issue "
            "so we can help?"
        )

    return response


def _build_evidence_block(historical_responses):
    """
    Convert retrieved historical examples into a compact evidence block.
    """
    evidence_lines = []

    for index, item in enumerate(historical_responses, start=1):
        if not isinstance(item, dict):
            continue

        customer = _clean_text(item.get("customer_message", ""))
        support = _clean_text(item.get("support_response", ""))

        if not customer and not support:
            continue

        score = item.get("score")

        if isinstance(score, (int, float)):
            score_text = f"{float(score):.4f}"
        else:
            score_text = "unknown"

        evidence_lines.append(
            f"EXAMPLE {index}\n"
            f"Retrieval score: {score_text}\n"
            f"Customer: {customer}\n"
            f"AppleSupport: {support}"
        )

    return "\n\n".join(evidence_lines)


def _parse_json(content):
    """
    Parse JSON even if the model accidentally wraps it in a markdown block.
    """
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)

        if not match:
            raise

        return json.loads(match.group(0))


def _call_model(customer_message, intent, historical_responses):
    evidence = _build_evidence_block(historical_responses)

    system_prompt = """
You are a customer-support response drafter for AppleSupport.

Your job is NOT to solve the customer's problem using your own knowledge.

Your ONLY source of truth is the historical AppleSupport examples supplied
in the user message.

You may:
- reuse a historical response,
- lightly paraphrase a historical response,
- combine wording from the supplied historical responses when both support
  the same action,
- ask for clarification when the evidence does not contain enough information.

You MUST NOT:
- use general knowledge about Apple products,
- invent troubleshooting steps,
- invent settings paths,
- invent buttons, menus, commands, or procedures,
- invent policies,
- invent eligibility rules,
- invent URLs,
- invent articles,
- invent support channels,
- invent availability information,
- invent locations,
- invent refund or warranty information,
- add technical instructions that are absent from the evidence.

Very important:
If the historical examples only ask the customer for more information,
your response must also ask for more information.

For example, if the evidence says:
"Could you tell us which device you're using?"

Do NOT turn it into:
"Go to Settings > General > About."

The second response contains information that was not supplied by the
historical evidence and is therefore forbidden.

For how-to/settings questions, be especially conservative. If the evidence
does not contain an actual step-by-step instruction, ask a clarification
question instead of providing steps from your own knowledge.

For transaction, repair, return, warranty, account, or security issues,
do not invent policy or eligibility information.

If a historical response contains a URL, you may include that URL only if
you reproduce the exact URL from the evidence.

The response should sound natural and concise, like a real AppleSupport
reply.

Return ONLY valid JSON with exactly these fields:

{
  "response": "string",
  "evidence_sufficient": true,
  "source_rank": 1
}

Rules for the fields:
- response: the proposed customer-facing reply.
- evidence_sufficient: true only when the response is directly supported
  by the supplied historical evidence. Otherwise false.
- source_rank: the main historical example used, starting at 1.
""".strip()

    user_prompt = f"""
Customer message:
{customer_message}

Predicted intent:
{intent}

Historical AppleSupport evidence:
{evidence}

Draft a response using ONLY this evidence.

Remember:
Do not solve the problem from your own knowledge.
Do not add instructions that are not present in the evidence.
If the evidence is insufficient, ask a clarification question.

Return JSON only.
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        reasoning_effort="low",
        temperature=0,
        max_completion_tokens=400,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Groq returned an empty response.")

    return _parse_json(content)


def generate_grounded_response(
    customer_message,
    intent,
    historical_responses,
):
    """
    Generate a historically grounded AppleSupport response.

    Safety strategy:
    1. Ask the LLM to paraphrase only supplied evidence.
    2. Require structured JSON.
    3. Reject unsupported instructions.
    4. Reject responses with almost no lexical relationship to evidence.
    5. Fall back to the highest-ranked historical response.

    This makes the generator fail safely: when evidence is weak, it prefers
    an existing AppleSupport response over invented troubleshooting advice.
    """

    historical_responses = historical_responses or []

    # No evidence means we should not ask the model to invent anything.
    if not historical_responses:
        return _safe_fallback(historical_responses)

    fallback = _safe_fallback(historical_responses)

    try:
        result = _call_model(
            customer_message=customer_message,
            intent=intent,
            historical_responses=historical_responses,
        )
    except Exception:
        return fallback

    if not isinstance(result, dict):
        return fallback

    generated = result.get("response")

    if not isinstance(generated, str):
        return fallback

    generated = _clean_text(generated)

    if not generated:
        return fallback

    evidence_sufficient = result.get("evidence_sufficient")

    # The model itself said that it could not ground the response.
    if evidence_sufficient is False:
        return fallback

    # Only use the main retrieved historical response for the hard grounding
    # checks. This prevents unrelated lower-ranked examples from legitimising
    # invented content.
    top_item = historical_responses[0]

    if isinstance(top_item, dict):
        top_historical_response = _clean_text(
            top_item.get("support_response", "")
        )
    else:
        top_historical_response = _clean_text(str(top_item))

    if not top_historical_response:
        return fallback

    # Reject concrete instructions that were not present in the main evidence.
    if _contains_unsupported_instruction(
        generated,
        top_historical_response,
    ):
        return fallback

    # Reject responses that are almost completely unrelated to the evidence.
    similarity = _evidence_similarity(
        generated,
        top_historical_response,
    )

    if similarity < 0.15:
        return fallback

    # Remove accidental surrounding whitespace/formatting one final time.
    return generated
# Backward-compatible function used by customer_support_agent.py
def generate_response(
    customer_message,
    historical_examples,
    intent=None,
):
    """
    Backward-compatible wrapper used by customer_support_agent.py.

    The agent expects a dictionary containing at least:
        response
    """

    if intent is None:
        intent = "UNKNOWN"

    response = generate_grounded_response(
        customer_message=customer_message,
        intent=intent,
        historical_responses=historical_examples,
    )

    return {
        "response": response,
    }