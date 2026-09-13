\# Golden Evaluation Set — Sampling and Labelling Note



\## Dataset

The golden evaluation set was created from the Customer Support on Twitter dataset, using customer messages directed to AppleSupport.

## Sampling

An initial sample of 500 AppleSupport customer messages was created and labeled for use as the training and holdout sets (`intent_train.csv` and `intent_test.csv`).

To ensure a strict, unbiased evaluation with **zero data leakage**, a completely separate random sample of 200 AppleSupport messages was generated. These 200 messages are **strictly disjoint** from the 500-example training/test pool. 

This final independent set of 200 examples satisfies the assignment requirement of 150–250 examples.

## Labelling

Each customer message was assigned exactly one primary intent using the AppleSupport-specific intent taxonomy defined in `docs/intent_taxonomy.md`.

The taxonomy contains 11 intents:

- I1 — Software / iOS / macOS Issue
- I2 — Battery / Charging / Power
- I3 — App / App Store / App Update Issue
- I4 — Connectivity / Network
- I5 — Messages / Communication
- I6 — Device / Hardware Issue
- I7 — Account / iCloud / Security
- I8 — Apple Services / Media
- I9 — Purchase / Order / Repair / Warranty
- I10 — How-to / Settings / Information
- I11 — Other / Unclear

The labelling process used the message's primary customer problem rather than individual keywords. For messages containing multiple symptoms, the main problem was selected. Messages with insufficient standalone context were labelled I11.

Each labelled example also contains a label confidence (usually "high") and a short human-written reason explaining *why* the assigned intent was chosen based on the context of the tweet.

## Final Golden Set

The final set contains:
- 200 examples
- Disjoint from the training set (no contamination)

## Important evaluation note

This golden evaluation set serves as the true held-out benchmark for the system's intent classification capabilities. Because it is completely disjoint from the training data, performance on this set accurately reflects the model's ability to generalize to unseen AppleSupport messages.
