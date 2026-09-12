\# CSA — Customer Support Agent



An evidence-based customer support agent built from the Customer Support on Twitter dataset.



The system focuses on \*\*AppleSupport\*\* and performs three tasks:



1\. Classifies a customer message into a small intent taxonomy.

2\. Retrieves historically similar AppleSupport conversations and drafts a grounded response.

3\. Decides whether the case should be handled automatically or escalated to a human.



The main design principle is:



> \*\*Use historical AppleSupport behavior as evidence instead of relying on general product knowledge.\*\*



\---



\## 1. Problem



Customer-support conversations contain noisy language, short messages, multiple symptoms, and multi-turn interactions.



The agent therefore needs to answer three questions:



\### Intent

What type of problem is the customer reporting?



\### Response

How has AppleSupport historically handled similar problems?



\### Escalation

Can the case be handled automatically, or should it go to a human?



\---



\## 2. Dataset



The project uses the \*\*Customer Support on Twitter\*\* dataset from Kaggle.



The dataset contains customer-to-brand and brand-to-customer tweets from many companies.



For this project, the selected brand is:



\*\*AppleSupport\*\*



The AppleSupport subset contains:



\- 106,860 AppleSupport tweets

\- 226,755 connected tweets

\- 74,613 reconstructed customer-rooted threads

\- 20,610 clean multi-turn alternating conversations

\- 92,842 customer/support response pairs used as the historical response pool



The full dataset is not required to reproduce the headline evaluation.



\---



\## 3. System Pipeline



```text

Customer message

&#x20;      |

&#x20;      v

+--------------------+

| Intent Classifier   |

| TF-IDF + Logistic   |

| Regression + rules  |

+--------------------+

&#x20;      |

&#x20;      v

+--------------------+

| Semantic Retrieval  |

| MiniLM embeddings   |

+--------------------+

&#x20;      |

&#x20;      v

Historical AppleSupport

response examples

&#x20;      |

&#x20;      v

+--------------------+

| Grounded Response   |

| Groq LLM generation |

+--------------------+

&#x20;      |

&#x20;      +----------------------+

&#x20;      |                      |

&#x20;      v                      v

&#x20;Response draft        Escalation decision

&#x20;                             |

&#x20;                             v

&#x20;                      Auto-handle / Human

