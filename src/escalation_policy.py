def decide_escalation(
    intent,
    retrieval_score,
    historical_responses
):
    """
    Decide whether the customer request should be escalated.

    Escalation is based on:
    1. Strength of retrieved historical evidence.
    2. Account/security intent.
    3. Purchase/repair/return/warranty intent.

    A historical DM instruction alone is NOT treated as proof
    that the current request requires human escalation.
    """

    # ---------------------------------------------------------
    # 1. Weak historical evidence
    # ---------------------------------------------------------
    if retrieval_score < 0.35:

        return {
            "escalate": True,
            "reason": (
                "Retrieved historical evidence is too weak."
            )
        }

    # ---------------------------------------------------------
    # 2. Account / iCloud / Security
    # ---------------------------------------------------------
    if intent == "I7":

        return {
            "escalate": True,
            "reason": (
                "Account or security cases may require "
                "human handling."
            )
        }

    # ---------------------------------------------------------
    # 3. Purchase / Order / Repair / Return / Warranty
    # ---------------------------------------------------------
    if intent == "I9":

        return {
            "escalate": True,
            "reason": (
                "Purchase, repair, return, or warranty "
                "cases may require human handling."
            )
        }

    # ---------------------------------------------------------
    # 4. Otherwise, auto-handle
    # ---------------------------------------------------------
    return {
        "escalate": False,
        "reason": (
            "Historical evidence supports an automated response."
        )
    }