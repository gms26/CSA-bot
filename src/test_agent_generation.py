import sys
sys.path.insert(0, ".")

from src.customer_support_agent import CustomerSupportAgent

agent = CustomerSupportAgent()


test_messages = [
    "My iPhone battery is draining very quickly.",
    "I cannot sign into my Apple ID.",
    "My iPhone cannot connect to Wi-Fi.",
    "My screen stopped responding.",
    "I cannot download apps from the App Store.",
    "My messages are not being delivered.",
    "How do I change my iPhone settings?",
    "My Apple Music songs disappeared.",
    "I need to return my iPhone.",
    "My iPhone keeps freezing after the update.",
]


for index, message in enumerate(test_messages, start=1):
    print()
    print("=" * 70)
    print(f"TEST CASE {index}")
    print("=" * 70)
    print("Customer:", message)

    result = agent.handle(message)

    print("Intent:", result["intent"])
    print("Response:", result["response"])
    print("Escalate:", result["escalate"])
    print("Reason:", result["escalation_reason"])

    print()
    print("Top retrieved example:")
    if result["retrieved_examples"]:
        top = result["retrieved_examples"][0]
        print("Similarity:", f"{top['similarity']:.4f}")
        print("Customer:", top["customer_message"])
        print("Support:", top["support_response"])