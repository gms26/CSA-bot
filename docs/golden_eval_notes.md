\# Golden Evaluation Set — Sampling and Labelling Note



\## Dataset



The golden evaluation set was created from the Customer Support on Twitter

dataset, using customer messages directed to AppleSupport.



\## Sampling



An initial sample of 500 AppleSupport customer messages was created using a

fixed random seed (`random\_state=42`). These 500 messages were manually

reviewed and labelled.



For the final golden evaluation set, 200 of the manually labelled examples

were selected using the same fixed random seed (`random\_state=42`).



The final set therefore contains 200 hand-labelled examples, satisfying the

assignment requirement of 150–250 examples.



\## Labelling



Each customer message was assigned exactly one primary intent using the

AppleSupport-specific intent taxonomy defined in `docs/intent\_taxonomy.md`.



The taxonomy contains 11 intents:



\- I1 — Software / iOS / macOS Issue

\- I2 — Battery / Charging / Power

\- I3 — App / App Store / App Update Issue

\- I4 — Connectivity / Network

\- I5 — Messages / Communication

\- I6 — Device / Hardware Issue

\- I7 — Account / iCloud / Security

\- I8 — Apple Services / Media

\- I9 — Purchase / Order / Repair / Warranty

\- I10 — How-to / Settings / Information

\- I11 — Other / Unclear



The labelling process used the message's primary customer problem rather than

individual keywords. For messages containing multiple symptoms, the main

problem was selected. Messages with insufficient standalone context were

labelled I11.



Each labelled example also contains a label confidence and a short reason

explaining the assigned intent.



\## Final Golden Set



The final set contains:



\- 200 examples

\- 11 represented intents

\- 0 missing intent labels



The intent distribution is:



| Intent | Examples |

|---|---:|

| I1 | 74 |

| I10 | 10 |

| I11 | 50 |

| I2 | 12 |

| I3 | 12 |

| I4 | 5 |

| I5 | 4 |

| I6 | 6 |

| I7 | 8 |

| I8 | 13 |

| I9 | 6 |

| \*\*Total\*\* | \*\*200\*\* |



\## Important evaluation note



The 200 examples were selected from a larger manually labelled sample of

500 examples. The golden labels were created before using this final subset

for evaluation. The set is intended as a held-out evaluation artifact for

the project and should not be used to train the final classifier.

