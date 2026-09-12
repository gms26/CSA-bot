import sys

sys.path.insert(0, ".")

from src.groq_client import client, MODEL_NAME


result = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "Reply with exactly: HELLO",
        },
        {
            "role": "user",
            "content": "Say HELLO.",
        },
    ],
    temperature=0,
    max_tokens=300,
)

print("=" * 70)
print("MODEL:", MODEL_NAME)
print("=" * 70)

print("CHOICES:", len(result.choices))

if result.choices:
    choice = result.choices[0]

    print("FINISH REASON:", choice.finish_reason)
    print("CONTENT:", repr(choice.message.content))
    print("MESSAGE:", choice.message)
else:
    print("No choices returned.")

print("=" * 70)
print("FULL RESPONSE:")
print(result)