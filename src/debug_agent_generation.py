import sys

sys.path.insert(0, ".")

from src.semantic_retriever import SemanticRetriever
from src.groq_client import client, MODEL_NAME


retriever = SemanticRetriever()

customer_message = "My iPhone battery is draining very quickly."

retrieved = retriever.retrieve(
    customer_message,
    top_k=5,
)

examples_text = "\n\n".join(
    [
        f"Historical example {i + 1}:\n"
        f"Customer: {example['customer_message']}\n"
        f"AppleSupport: {example['support_response']}"
        for i, example in enumerate(retrieved)
    ]
)

prompt = f"""
You are drafting a customer-support response for AppleSupport.

Customer message:
{customer_message}

Historical AppleSupport examples:

{examples_text}

Draft a concise customer-facing response based only on these
historical examples.

Do not invent solutions, policies, URLs, or product facts.

Return ONLY valid JSON in this exact format:

{{
  "response": "short customer-facing response",
  "escalate": true,
  "escalation_reason": "short reason"
}}
"""

print("=" * 70)
print("TESTING ACTUAL AGENT-STYLE GROQ REQUEST")
print("=" * 70)

result = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": (
                "Return exactly one valid JSON object. "
                "Do not return an empty response."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
    temperature=0.1,
    max_tokens=300,
)

print()
print("FINISH REASON:", result.choices[0].finish_reason)
print("CONTENT:", repr(result.choices[0].message.content))
print("REASONING:", repr(result.choices[0].message.reasoning))
print()
print("FULL RESPONSE:")
print(result)