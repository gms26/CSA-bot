# CSA - Customer Support Agent

An evidence-based customer support agent built from the **Customer Support on Twitter** dataset.

The system focuses on **AppleSupport** and performs three tasks:

1. Classifies an incoming customer message into a small intent taxonomy.
2. Retrieves historically similar AppleSupport conversations and drafts a grounded response.
3. Decides whether the case should be handled automatically or escalated to a human.

> **Design principle:** Use historical AppleSupport behavior as evidence instead of relying on general product knowledge.

---

## 1. Problem Framing

Customer-support conversations contain noisy language, short messages, multiple symptoms, incomplete context, and multi-turn interactions.

The agent therefore needs to answer three questions:

### Intent

What type of problem is the customer reporting?

### Response

How has AppleSupport historically handled similar problems?

### Escalation

Can the case be handled automatically, or should it go to a human?

### What good means

A useful support agent should:

- identify the customer's main intent reasonably well;
- retrieve examples addressing the same underlying problem;
- avoid inventing troubleshooting steps or policies;
- produce a response consistent with historical support behavior;
- escalate cases where automated handling is inappropriate.

### What is not built

This is a take-home evaluation prototype, not a production support system.

It does not include:

- live Apple support systems;
- real customer account access;
- order or repair databases;
- production authentication;
- live policy verification;
- a production web application;
- continuous online learning;
- full-dataset production-scale inference.

---

## 2. Dataset

The project uses the **Customer Support on Twitter** dataset from Kaggle.

The dataset contains customer-to-brand and brand-to-customer tweets from many companies.

The selected brand is:

**AppleSupport**

The AppleSupport subset used during development contains:

- 106,860 AppleSupport tweets
- 226,755 connected tweets
- 74,613 reconstructed customer-rooted threads
- 20,610 clean multi-turn alternating conversations
- 92,842 customer/support response pairs used as the historical response pool

The full Kaggle dataset is not committed to GitHub.

The repository uses a subsample/workflow so the project can be reproduced without committing the full dataset.

---

## 3. System Pipeline

```text
Customer message
       |
       v
+----------------------+
| Intent Classifier     |
| TF-IDF + Logistic     |
| Regression + rules    |
+----------------------+
       |
       v
+----------------------+
| Semantic Retrieval    |
| all-MiniLM-L6-v2      |
+----------------------+
       |
       v
Historical AppleSupport
response examples
       |
       v
+----------------------+
| Grounded Response     |
| Groq GPT-OSS-20B      |
+----------------------+
       |
       +----------------------+
       |                      |
       v                      v
 Response draft        Escalation decision
                              |
                              v
                       Auto-handle / Human

