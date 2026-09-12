# Retrieval Relevance Labeling Prompt

You are evaluating a customer-support retrieval system.

For each evaluation query, there are 5 retrieved historical customer messages.

Your task is to determine whether each retrieved message is relevant to the query.

## Definition

Label a candidate:

- 1 = Relevant
  The candidate clearly addresses the SAME underlying customer problem,
  request, or issue as the query.

- 0 = Not relevant
  The candidate does not clearly address the same underlying problem.

## Important rules

1. Do NOT judge relevance only from shared words.
2. Same device, operating system, Apple product, or general topic is NOT enough.
3. If the query is incomplete or context-dependent and the candidate cannot
   be confirmed as addressing the same problem, label 0.
4. Different underlying problems = 0.
5. Closely related symptoms of the same problem can be 1.
6. A candidate does not need to use exactly the same words as the query.
7. Do not invent missing context.
8. When uncertain, choose 0.
9. Judge the customer message itself, not whether the historical support
   response is a good answer.
10. Give one label for every candidate.

## Examples

Example 1:

Query:
"Newest one"

Candidate:
"Newest"

Label:
0

Reason:
Both messages are vague and context-dependent. The underlying problem cannot
be confirmed.

---

Example 2:

Query:
"Why does my phone say No Sim fix this"

Candidate:
"My iPhone says Invalid SIM"

Label:
1

Reason:
Both clearly describe the same SIM problem.

---

Example 3:

Query:
"my battery is draining really fast"

Candidate:
"my iPhone battery dies twice a day"

Label:
1

Reason:
Both describe excessive battery drain.

---

Example 4:

Query:
"my battery is draining really fast"

Candidate:
"my Wi-Fi keeps disconnecting"

Label:
0

Reason:
The underlying problems are different.

---

Example 5:

Query:
"my iPhone is slow after iOS 11"

Candidate:
"iOS 11 made my iPhone extremely slow"

Label:
1

Reason:
Both describe iPhone slowness associated with iOS 11.

---

Example 6:

Query:
"my iPhone is slow after iOS 11"

Candidate:
"iOS 11 has a problem with my battery"

Label:
0

Reason:
Both mention iOS 11, but the underlying problems are different.

## Output format

For every query, return exactly 5 results in this format:

Q<number>:
1. <0 or 1> - <short reason>
2. <0 or 1> - <short reason>
3. <0 or 1> - <short reason>
4. <0 or 1> - <short reason>
5. <0 or 1> - <short reason>

Do not skip any candidate.