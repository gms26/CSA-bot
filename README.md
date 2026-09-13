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

The selected brand is: **AppleSupport**

The AppleSupport subset used during development contains:
- 106,860 AppleSupport tweets
- 226,755 connected tweets
- 74,613 reconstructed customer-rooted threads
- 20,610 clean multi-turn alternating conversations
- 92,842 customer/support response pairs used as the historical response pool

The full Kaggle dataset is not committed to GitHub. The repository uses a subsample/workflow so the project can be reproduced without committing the full dataset.

---

## 3. System Pipeline

```text
Customer message
       |
       v
+----------------------+
| Intent Classifier    |
| TF-IDF + Logistic    |
| Regression + rules   |
+----------------------+
       |
       v
+----------------------+
| Semantic Retrieval   |
| all-MiniLM-L6-v2     |
+----------------------+
       |
       v
Historical AppleSupport
response examples
       |
       v
+----------------------+
| Grounded Response    |
| Groq GPT-OSS-20B     |
+----------------------+
       |
       +----------------------+
       |                      |
       v                      v
 Response draft        Escalation decision
                              |
                              v
                       Auto-handle / Human
```

---

## 4. Quick Start & Reproducibility

### Setup
1. Clone the repository and navigate to the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the environment variables:
   Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

### Dataset Download
To run the full pipeline, download the **Customer Support on Twitter** dataset from Kaggle and place it at `raw/twcs/twcs.csv`.

### Run Commands
To run the end-to-end pipeline (data extraction, thread reconstruction, evaluation):
```bash
python src/extract_apple_data.py
python src/reconstruct_apple_threads.py
python src/create_response_pairs.py
python src/clean_response_pairs.py
python src/sample_customer_messages.py
# (Manual labeling of 500 examples occurs here)
python src/create_intent_holdout.py
python src/customer_support_agent.py  # Smoke test
```
To run the evaluation scripts:
```bash
python src/evaluate_intent_baseline.py
python src/evaluate_holdout_baseline.py
python src/evaluate_hybrid_intent.py
python src/evaluate_agent_intent.py
python src/evaluate_retrieval_metrics.py
python src/evaluate_final_response_judge.py
```

---

## 5. Evaluation Results & Baselines

### Intent Classification
| System | Implementation | Metric (Accuracy) |
|---|---|---|
| **Trivial baseline** | Majority class (I1) | 34.20% |
| **Simple baseline** | TF-IDF + LogReg (100 test holdout) | 48.00% |
| **Final system** | TF-IDF + LogReg + Rules (CV on 500) | 57.20% |

### Semantic Retrieval Evaluation (100 Queries)
| Metric | Value |
|---|---|
| Recall@1 | 79.0% |
| Recall@3 | 83.0% |
| Recall@5 | 83.0% |
| Precision@5 | 76.6% |
| MRR | 0.807 |
*Note: Retrieval labels were manually verified with LLM-assistance.*

### LLM Judge Evaluation (20 Responses)
| Dimension | LLM Judge Score |
|---|---|
| Relevance | 100% (20/20) |
| Groundedness | 95% (19/20) |
| Safety | 100% (20/20) |
| Escalation Correctness | 80% (16/20) |

---

## 6. Golden Evaluation Set

The golden evaluation set is located at `docs/golden_eval.csv` and contains 200 manually labelled examples, ensuring the assignment requirement of 150-250 examples is met. The methodology includes assigning exactly one primary intent out of 11 possible intents based on the customer's main problem. Refer to `docs/golden_eval_notes.md` for more details.

---

## 7. Human–LLM Agreement

Evaluation on 20 frozen agent-response cases:
| Dimension | Agreement |
|---|---:|
| Relevance | 90.0% (18/20) |
| Groundedness | 90.0% (18/20) |
| Safety | 75.0% (15/20) |
| Escalation correctness | 95.0% (19/20) |

Human review was completed independently from the LLM judge scores.

---

## 8. Failure Analysis

Based on the evaluation artifacts, here are the top 5 failure modes:

1. **Encoding mojibake in generation**: 
   - *Example*: Generated response contains "â€™" instead of apostrophes (e.g., "youâ€™re").
   - *Hypothesis*: The dataset contains raw Unicode anomalies that were not fully stripped before hitting the LLM or being saved.
2. **Over-escalation**:
   - *Example*: The LLM judge scored 80% on escalation correctness, while human review scored 15/20. The agent escalates cases that it could have easily handled automatically.
   - *Hypothesis*: The rule-based escalation policy aggressively escalates any I7 or I9 intent without accounting for easily solvable sub-intents.
3. **Invented troubleshooting steps**:
   - *Example*: (Eval ID 8) "try these steps: 1) Connect your phone to a charger and hold the power button..."
   - *Hypothesis*: The LLM hallucinates technical steps that were not present in the historical evidence despite prompt guardrails.
4. **Vague messages classified as I11**:
   - *Example*: Messages like "Newest one" get classified as I11 (Other/Unclear) and receive a generic clarification response, even though context might exist in prior messages.
   - *Hypothesis*: The system processes messages statelessly, losing thread-level context.
5. **Intent misclassification for multi-symptom issues**:
   - *Example*: Users mentioning both "battery draining" and "IOS update lagging" get classified poorly due to mixed signals.
   - *Hypothesis*: TF-IDF struggles to weigh the primary issue correctly over keyword frequency.

---

## 9. What is misleading about my headline number?

**Golden Set Contamination:**
The headline accuracy is inherently inflated because the 200 golden evaluation examples are a strict subset of the 500-sample used to construct the `intent_train.csv` (163 of 200 examples overlap). Since the classifier was trained on this overlapping data, evaluating on the golden set introduces a serious data leakage issue.

**Small Sample Sizes:**
The LLM response judge and human agreement evaluations only use 20 examples (N=20). High percentage scores (like 100% Relevance) are statistically noisy and may not reflect production reliability. Furthermore, the Human-LLM agreement uses plain percentage agreement rather than Cohen's kappa, overestimating reliability given high base rates for positive classes.

**Retrieval Labels are LLM-Assisted:**
The retrieval recall metrics rely on labels that were primarily LLM-assisted, meaning the retrieval numbers reflect LLM alignment rather than strict human judgment.

---

## 10. What I would do with one more week

1. **Fix Data Leakage**: Disentangle the golden evaluation set from the training set by labeling an entirely disjoint sample of 250 held-out cases to give a true representation of intent accuracy.
2. **Expand the Response Evaluation Set**: Scale the response evaluation from N=20 to N=100+ to gain statistically significant insights into hallucination rates and escalation policy efficacy.
3. **Advanced RAG Pipeline**: Replace naive cosine similarity with a cross-encoder reranker for historical evidence to improve precision@1.
4. **Context-Aware Intent Classification**: Pass the prior conversation turn to the classifier to properly resolve anaphoric or vague responses (e.g., "Yes, I did").
5. **Clean Text Processing**: Implement rigorous encoding standardisation (e.g., fixing ftfy mojibake) before creating the embeddings or prompting the LLM.

---

## 11. Decision Log

1. **Brand Selection (AppleSupport)**: Chose AppleSupport due to the massive volume of multi-turn threads available in the dataset.
2. **Thread Reconstruction via Earliest Child**: Reconstructed multi-turn threads by always following the chronologically first response to avoid branching ambiguity.
3. **Adjacent Customer-Support Pairs**: Only used strictly adjacent messages for the retrieval pool to ensure the support response explicitly answers the immediate customer query.
4. **11-Intent Taxonomy**: Designed based on empirical clusters in the Apple data (e.g., separating hardware I6 from software I1) rather than a priori guesses.
5. **Hybrid Classifier (TF-IDF + Rules)**: Used TF-IDF for broad classification but overrode predictions with hardcoded keyword rules for critical intents (like connectivity).
6. **Intent-aware Reranking**: Boosted retrieval candidates that match the predicted intent of the incoming message to reduce off-topic retrieval.
7. **Same-Thread Exclusion**: Explicitly excluded responses from the same thread during retrieval evaluation to prevent direct data leakage.
8. **all-MiniLM-L6-v2**: Used a lightweight local embedding model to ensure processing 92K vectors was feasible on consumer hardware without excessive API costs.
9. **Evidence-Only Prompting**: Instructed the LLM to strictly paraphrase historical support responses rather than invent solutions to prevent harmful hallucinations.
10. **Fallback to Historical Response**: Designed the agent to gracefully return a raw historical response if the LLM output failed safety checks or hallucination guards.
11. **Rule-Based Escalation**: Escalation is hardcoded for sensitive intents (I7 Security, I9 Purchase) to guarantee human handling of high-risk cases.
12. **N=20 Response Evaluation**: Frozen the response evaluation pool to 20 examples to allow manual human verification against the LLM judge.
13. **Groq / GPT-OSS-20B**: Selected for extremely fast inference during generation without sacrificing instruction-following capability.

---

## 12. Limitations
- Does not support stateful multi-turn interactions.
- Cannot process images or media.
- Evaluation metrics rely on heavily downsampled datasets due to resource constraints.

---

## 13. Attribution
- **Dataset**: Customer Support on Twitter (Kaggle)
- **Embeddings**: `sentence-transformers` library
- **LLM**: Groq API
