import re

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.intent_classifier import IntentClassifier
from src.semantic_retriever import SemanticRetriever
from src.escalation_policy import decide_escalation
from src.generate_grounded_response import generate_grounded_response


HISTORICAL_PAIRS_PATH = (
    "processed/apple_support_response_pairs_clean.csv"
)


def _clean_response(text):
    """Clean minor formatting issues without changing response meaning."""

    if not text:
        return ""

    text = str(text).strip()

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text,
    )

    replacements = {
        r"\bwith\s*which\b": "with which",
        r"\bwithwhich\b": "with which",
        r"\bhuman\s*handling\b": "human handling",
        r"\bhumanhandling\b": "human handling",
        r"\bAre\s*you\b": "Are you",
        r"\bAreyou\b": "Are you",
        r"\bare\s*you\b": "are you",
        r"\bareyou\b": "are you",
        r"\blook\s*into\b": "look into",
        r"\blookinto\b": "look into",
        r"\bmore\s*details\b": "more details",
        r"\bmoredetails\b": "more details",
        r"\bDirect\s*Message\b": "Direct Message",
        r"\bDirectMessage\b": "Direct Message",
        r"\bupgrade\s*program\b": "upgrade program",
        r"\bupgradeprogram\b": "upgrade program",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    cleaned_sentences = []
    seen = set()

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        normalized = re.sub(
            r"[^a-z0-9 ]",
            "",
            sentence.lower(),
        ).strip()

        if (
            normalized
            and normalized not in seen
        ):
            cleaned_sentences.append(
                sentence
            )
            seen.add(
                normalized
            )

    return " ".join(
        cleaned_sentences
    ).strip()


def _ensure_escalation_consistency(
    response,
    escalate,
):
    """Ensure escalated cases contain a private support handoff."""

    response = _clean_response(
        response
    )

    if escalate:

        lower = response.lower()

        if (
            "dm" not in lower
            and "direct message" not in lower
        ):

            if (
                response
                and response[-1] not in ".!?"
            ):
                response += "."

            response += (
                " Please DM us so we can look into this with you."
            )

    return response.strip()


def _get_historical_intent(
    classifier,
    message,
):
    """Classify a historical customer message."""

    try:

        result = classifier.predict(
            message
        )

        if isinstance(
            result,
            dict,
        ):
            return result.get(
                "intent"
            )

        return result

    except Exception:

        return None


def _detect_transaction_actions(
    message,
):
    """
    Detect transaction/support actions from a customer message.

    Return is deliberately restricted to product-return language.
    """

    text = str(
        message
    ).lower()

    actions = set()

    # =============================================================
    # Product names
    # =============================================================

    product_pattern = (
        r"(?:iphone|ipad|ipod|mac|macbook|imac|"
        r"device|product|item|phone|computer)"
    )

    # =============================================================
    # RETURN
    # =============================================================

    return_patterns = [

        # return my iPhone
        rf"\breturn\s+"
        rf"(?:my|the|this|an|a)\s+"
        rf"{product_pattern}\b",

        # return iPhone
        rf"\breturn\s+"
        rf"{product_pattern}\b",

        # returned my iPhone
        rf"\breturned\s+"
        rf"(?:my|the|this)\s+"
        rf"{product_pattern}\b",

        # returning my iPhone
        rf"\breturning\s+"
        rf"(?:my|the|this)\s+"
        rf"{product_pattern}\b",

        # want to return the iPhone
        rf"\bwant(?:ing)?\s+to\s+return\s+"
        rf"(?:my|the|this|an|a)?\s*"
        rf"{product_pattern}\b",

        # need to return the iPhone
        rf"\bneed\s+to\s+return\s+"
        rf"(?:my|the|this|an|a)?\s*"
        rf"{product_pattern}\b",

        # can I return the iPhone
        rf"\bcan\s+i\s+return\s+"
        rf"(?:my|the|this|an|a)?\s*"
        rf"{product_pattern}\b",

        # any way I can return the iPhone
        rf"\bany\s+way\s+i\s+can\s+return\s+"
        rf"(?:my|the|this|an|a)?\s*"
        rf"{product_pattern}\b",

        # take back my iPhone
        rf"\btake\s+back\s+"
        rf"(?:my|the|this)\s+"
        rf"{product_pattern}\b",

        # return and re-buy
        r"\breturn\s+and\s+"
        r"(?:re-?buy|buy|replace)\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in return_patterns
    ):
        actions.add(
            "return"
        )

    # =============================================================
    # Refund
    # =============================================================

    refund_patterns = [
        r"\brefund\b",
        r"\brefunded\b",
        r"\brefunding\b",
        r"\bmoney\s+back\b",
        r"\bget\s+my\s+money\s+back\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in refund_patterns
    ):
        actions.add(
            "refund"
        )

    # =============================================================
    # Replacement
    # =============================================================

    replacement_patterns = [
        r"\breplace\b",
        r"\breplaced\b",
        r"\breplacement\b",
        r"\breplacing\b",
        r"\bget\s+a\s+replacement\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in replacement_patterns
    ):
        actions.add(
            "replace"
        )

    # =============================================================
    # Repair
    # =============================================================

    repair_patterns = [
        r"\brepair\b",
        r"\brepaired\b",
        r"\brepairing\b",
        r"\brepairs\b",
        r"\bneeds?\s+repair\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in repair_patterns
    ):
        actions.add(
            "repair"
        )

    # =============================================================
    # Warranty
    # =============================================================

    if re.search(
        r"\bwarranty\b",
        text,
    ):
        actions.add(
            "warranty"
        )

    # =============================================================
    # Purchase
    # =============================================================

    purchase_patterns = [
        r"\bpurchase\b",
        r"\bpurchased\b",
        r"\bbuy\b",
        r"\bbuying\b",
        r"\bbought\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in purchase_patterns
    ):
        actions.add(
            "purchase"
        )

    # =============================================================
    # Order
    # =============================================================

    order_patterns = [
        r"\border\b",
        r"\bordered\b",
        r"\bordering\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in order_patterns
    ):
        actions.add(
            "order"
        )

    # =============================================================
    # Delivery
    # =============================================================

    delivery_patterns = [
        r"\bdelivery\b",
        r"\bdelivered\b",
        r"\bshipping\b",
        r"\bshipment\b",
        r"\bshipped\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in delivery_patterns
    ):
        actions.add(
            "delivery"
        )

    # =============================================================
    # Exchange
    # =============================================================

    exchange_patterns = [
        r"\bexchange\b",
        r"\bexchanged\b",
        r"\bexchanging\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in exchange_patterns
    ):
        actions.add(
            "exchange"
        )

    return actions


def _action_match_score(
    current_message,
    historical_message,
):
    """
    Compare transaction actions between two customer messages.
    """

    current_actions = (
        _detect_transaction_actions(
            current_message
        )
    )

    historical_actions = (
        _detect_transaction_actions(
            historical_message
        )
    )

    if (
        not current_actions
        or not historical_actions
    ):
        return 0

    if (
        current_actions
        & historical_actions
    ):
        return 3

    return -3


class CustomerSupportAgent:

    def __init__(self):

        print(
            "Loading intent classifier..."
        )

        self.intent_classifier = (
            IntentClassifier()
        )

        print(
            "Loading semantic retriever..."
        )

        self.retriever = (
            SemanticRetriever()
        )

        print(
            "Loading historical response pairs..."
        )

        self.historical_pairs = (
            pd.read_csv(
                HISTORICAL_PAIRS_PATH
            )
        )

        self.historical_pairs[
            "customer_message"
        ] = (
            self.historical_pairs[
                "customer_message"
            ]
            .fillna("")
            .astype(str)
        )

        self.historical_pairs[
            "support_response"
        ] = (
            self.historical_pairs[
                "support_response"
            ]
            .fillna("")
            .astype(str)
        )

        print(
            f"Loaded "
            f"{len(self.historical_pairs):,} "
            "historical response pairs."
        )

    def _retrieve_i9_action_candidates(
        self,
        customer_message,
        max_candidates=100,
    ):
        """
        Retrieve historical transaction cases.

        For product returns, only historical messages that contain
        explicit product-return language are considered.
        """

        current_actions = (
            _detect_transaction_actions(
                customer_message
            )
        )

        if not current_actions:
            return []

        matches = []

        for row_index, row in (
            self.historical_pairs.iterrows()
        ):

            historical_customer = str(
                row[
                    "customer_message"
                ]
            )

            historical_actions = (
                _detect_transaction_actions(
                    historical_customer
                )
            )

            if not (
                current_actions
                & historical_actions
            ):
                continue

            matches.append(
                {
                    "row_index":
                        row_index,

                    "customer_message":
                        historical_customer,

                    "support_response":
                        str(
                            row[
                                "support_response"
                            ]
                        ),

                    "thread_id":
                        row.get(
                            "thread_id",
                            "",
                        ),

                    "action_score":
                        3,
                }
            )

        if not matches:
            return []

        # =============================================================
        # Semantic similarity
        # =============================================================

        query_embedding = (
            self.retriever.model.encode(
                [customer_message],
                normalize_embeddings=True,
            )
        )

        candidate_indices = [
            item[
                "row_index"
            ]
            for item in matches
        ]

        candidate_embeddings = (
            self.retriever.embeddings[
                candidate_indices
            ]
        )

        similarities = (
            cosine_similarity(
                query_embedding,
                candidate_embeddings,
            )[0]
        )

        results = []

        for item, similarity in zip(
            matches,
            similarities,
        ):

            similarity = float(
                similarity
            )

            results.append(
                {
                    "thread_id":
                        item[
                            "thread_id"
                        ],

                    "customer_message":
                        item[
                            "customer_message"
                        ],

                    "support_response":
                        item[
                            "support_response"
                        ],

                    "similarity":
                        similarity,

                    "original_similarity":
                        similarity,

                    "historical_intent":
                        None,

                    "action_score":
                        item[
                            "action_score"
                        ],

                    "combined_score":
                        similarity
                        + 0.60,

                    "retrieval_source":
                        "action_match",
                }
            )

        results.sort(
            key=lambda item:
                item[
                    "original_similarity"
                ],
            reverse=True,
        )

        return results[
            :max_candidates
        ]

    def _retrieve_with_intent_awareness(
        self,
        customer_message,
        intent,
        top_k=5,
    ):
        """
        Retrieve historical examples and rerank them using intent.

        I9 transaction cases receive a strong preference when the
        historical customer message contains the same transaction action.
        """

        candidates = (
            self.retriever.retrieve(
                customer_message,
                top_k=50,
            )
        )

        if intent == "I9":

            action_candidates = (
                self._retrieve_i9_action_candidates(
                    customer_message,
                    max_candidates=100,
                )
            )

            candidates.extend(
                action_candidates
            )

        # =============================================================
        # Deduplicate
        # =============================================================

        unique_candidates = {}

        for candidate in candidates:

            customer_text = candidate.get(
                "customer_message",
                candidate.get(
                    "customer",
                    "",
                ),
            )

            support_text = candidate.get(
                "support_response",
                candidate.get(
                    "support",
                    "",
                ),
            )

            key = (
                customer_text.strip().lower(),
                support_text.strip().lower(),
            )

            if key not in unique_candidates:

                unique_candidates[
                    key
                ] = candidate

        candidates = list(
            unique_candidates.values()
        )

        # =============================================================
        # Rerank
        # =============================================================

        reranked = []

        current_actions = (
            _detect_transaction_actions(
                customer_message
            )
        )

        for candidate in candidates:

            historical_customer = (
                candidate.get(
                    "customer_message",
                    candidate.get(
                        "customer",
                        "",
                    ),
                )
            )

            historical_intent = (
                _get_historical_intent(
                    self.intent_classifier,
                    historical_customer,
                )
            )

            similarity = float(
                candidate.get(
                    "original_similarity",
                    candidate.get(
                        "similarity",
                        candidate.get(
                            "score",
                            0.0,
                        ),
                    ),
                )
            )

            # Same intent boost.
            intent_boost = 0.0

            if (
                historical_intent
                == intent
            ):
                intent_boost = 0.15

            # Transaction action.
            action_score = 0

            if intent == "I9":

                action_score = (
                    _action_match_score(
                        customer_message,
                        historical_customer,
                    )
                )

            # =========================================================
            # I9 ranking
            # =========================================================

            if (
                intent == "I9"
                and current_actions
            ):

                if action_score > 0:

                    combined_score = (
                        similarity
                        + intent_boost
                        + 1.00
                    )

                else:

                    combined_score = (
                        similarity
                        + intent_boost
                        - 0.50
                    )

            else:

                combined_score = (
                    similarity
                    + intent_boost
                )

            enriched = dict(
                candidate
            )

            enriched[
                "historical_intent"
            ] = historical_intent

            enriched[
                "original_similarity"
            ] = similarity

            enriched[
                "action_score"
            ] = action_score

            enriched[
                "combined_score"
            ] = combined_score

            reranked.append(
                enriched
            )

        reranked.sort(
            key=lambda item:
                item[
                    "combined_score"
                ],
            reverse=True,
        )

        return reranked[
            :top_k
        ]

    def handle(
        self,
        customer_message,
    ):
        """Process one incoming customer message."""

        if (
            not customer_message
            or not str(
                customer_message
            ).strip()
        ):

            return {
                "intent":
                    "I11",

                "response":
                    (
                        "Could you please provide a little more detail "
                        "about the issue?"
                    ),

                "escalate":
                    True,

                "escalation_reason":
                    (
                        "The customer message is empty or unclear."
                    ),

                "retrieved_examples":
                    [],
            }

        customer_message = str(
            customer_message
        ).strip()

        # =============================================================
        # 1. Intent classification
        # =============================================================

        intent_result = (
            self.intent_classifier.predict(
                customer_message
            )
        )

        if isinstance(
            intent_result,
            dict,
        ):

            intent = (
                intent_result.get(
                    "intent",
                    "I11",
                )
            )

        else:

            intent = (
                intent_result
            )

        # =============================================================
        # 2. Retrieval
        # =============================================================

        retrieved_examples = (
            self._retrieve_with_intent_awareness(
                customer_message,
                intent,
                top_k=5,
            )
        )

        if retrieved_examples:

            retrieval_score = float(
                retrieved_examples[
                    0
                ].get(
                    "original_similarity",
                    retrieved_examples[
                        0
                    ].get(
                        "similarity",
                        0.0,
                    ),
                )
            )

        else:

            retrieval_score = 0.0

        # =============================================================
        # 3. Escalation
        # =============================================================

        escalation = (
            decide_escalation(
                intent,
                retrieval_score,
                retrieved_examples,
            )
        )

        escalate = bool(
            escalation.get(
                "escalate",
                False,
            )
        )

        escalation_reason = (
            escalation.get(
                "reason",
                "",
            )
        )

        # =============================================================
        # 4. Historical evidence
        # =============================================================

        historical_examples = []

        for example in (
            retrieved_examples
        ):

            historical_examples.append(
                {
                    "customer_message":
                        example.get(
                            "customer_message",
                            example.get(
                                "customer",
                                "",
                            ),
                        ),

                    "support_response":
                        example.get(
                            "support_response",
                            example.get(
                                "support",
                                "",
                            ),
                        ),

                    "similarity":
                        example.get(
                            "original_similarity",
                            example.get(
                                "similarity",
                                0.0,
                            ),
                        ),
                }
            )

        # =============================================================
        # 5. Grounded generation
        # =============================================================

        generation_result = (
            generate_grounded_response(
                customer_message,
                intent,
                historical_examples,
            )
        )

        if isinstance(
            generation_result,
            dict,
        ):

            response = (
                generation_result.get(
                    "response",
                    "",
                )
            )

        else:

            response = str(
                generation_result
            )

        response = _clean_response(
            response
        )

        # =============================================================
        # 6. Escalation consistency
        # =============================================================

        response = (
            _ensure_escalation_consistency(
                response,
                escalate,
            )
        )

        # =============================================================
        # 7. Fallback
        # =============================================================

        if not response:

            if retrieved_examples:

                response = (
                    retrieved_examples[
                        0
                    ].get(
                        "support_response",
                        (
                            "Could you provide a little more detail "
                            "so we can help?"
                        ),
                    )
                )

            else:

                response = (
                    "Could you provide a little more detail "
                    "about the issue so we can help?"
                )

            response = _clean_response(
                response
            )

        # =============================================================
        # 8. Return
        # =============================================================

        return {
            "intent":
                intent,

            "response":
                response,

            "escalate":
                escalate,

            "escalation_reason":
                escalation_reason,

            "retrieved_examples":
                retrieved_examples,
        }


if __name__ == "__main__":

    agent = (
        CustomerSupportAgent()
    )

    test_message = (
        "My iPhone battery is draining very quickly."
    )

    result = agent.handle(
        test_message
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST"
    )

    print(
        "=" * 70
    )

    print(
        "Customer:",
        test_message,
    )

    print(
        "Intent:",
        result[
            "intent"
        ],
    )

    print(
        "Response:",
        result[
            "response"
        ],
    )

    print(
        "Escalate:",
        result[
            "escalate"
        ],
    )

    print(
        "Reason:",
        result[
            "escalation_reason"
        ],
    )

    if result[
        "retrieved_examples"
    ]:

        top = result[
            "retrieved_examples"
        ][0]

        print(
            "\nTop retrieved example:"
        )

        print(
            "Similarity:",
            round(
                top.get(
                    "original_similarity",
                    0.0,
                ),
                4,
            ),
        )

        print(
            "Combined score:",
            round(
                top.get(
                    "combined_score",
                    0.0,
                ),
                4,
            ),
        )

        print(
            "Action score:",
            top.get(
                "action_score",
                0,
            ),
        )

        print(
            "Retrieval source:",
            top.get(
                "retrieval_source",
                "semantic",
            ),
        )

        print(
            "Customer:",
            top.get(
                "customer_message",
                "",
            ),
        )

        print(
            "Support:",
            top.get(
                "support_response",
                "",
            ),
        )