# CloudDocs RAG System

**[Try the live demo](https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/)**

---

Cloud documentation is scattered across three providers with overlapping terminology. Searching it manually is slow. Asking an AI directly is fast but unreliable — it makes up service names, quotes outdated pricing, and describes features that don't exist.

This project builds a system that reads real AWS, Azure, and GCP documentation before answering your question — so its answers are grounded in what the docs actually say, not what the AI guesses.

> **How it works in plain English:** The system scrapes cloud documentation, breaks it into passages, and stores them in a searchable database. When you ask a question, it finds the most relevant passages, hands them to an AI, and asks the AI to answer *using only what it just read*. The AI can't make things up because it's constrained to the retrieved text.

![RAG Chat Screenshot](ss/ss1.png)

---

## How it works

| Part | What it does |
| --- | --- |
| Scraper | Downloads AWS, Azure, and GCP documentation pages |
| Chunker | Splits pages into overlapping 1,000-character passages so context isn't lost at boundaries |
| Embedder | Converts each passage into a list of numbers that captures its *meaning* — similar topics get similar numbers |
| Database | Stores all passages with their meaning-numbers for fast similarity search |
| Retriever | When a question arrives, converts it the same way and finds the passages whose meaning is closest |
| AI model | Reads the retrieved passages and writes an answer — constrained to only what it was shown |

The AI model (Groq Llama 3.1 8B) is free to use and fast. A paid model like GPT-4o would produce better answers but isn't necessary to demonstrate the architecture.

---

## The journey

The system worked from day one — but "it seems to work" is not a measurement. The real work was building an honest test framework and running experiments to find out *why* it failed and *what actually fixed it*.

---

### 1. Build a test set and measure baseline

To measure whether the system was actually retrieving the right information, a test set of 27 questions was created — each with a known correct source document. Three things were measured:

- **Hit rate (Recall@K)** — did the right document appear in the top K results? (1.0 = perfect, 0.0 = never)
- **Rank quality (MRR@K)** — was the right document near the top, or buried? (rank 1 = score 1.0, rank 2 = 0.5, etc.)
- **Answer grounding (Faithfulness)** — did the AI's answer stick to what was retrieved, or did it add things that weren't there? Scored 1–5 by a second AI acting as judge.

Baseline results: **Hit rate = 0.630, Rank quality = 0.611, Grounding = 3.7/5**

10 out of 27 questions failed. Every single miss was an AWS question.

---

### 2. Find the right number of passages to retrieve

Before investigating *why* AWS failed, one key setting was tuned first: how many passages to show the AI. Too few and it might miss relevant information. Too many and it gets confused by noise.

| Passages shown | Hit rate | Grounding | Time to run |
| --- | --- | --- | --- |
| 1 | 0.593 | 3.85 | 133s |
| **3** | **0.630** | **4.11** | **308s** |
| 5 | 0.630 | 3.74 | 399s |
| 10 | 0.630 | 3.93 | 641s |

Hit rate stops improving after 3 passages. Showing more just increases processing time and slightly reduces answer quality. **3 passages became the system default.**

![K-Sweep Chart](eval/k_sweep_chart.png)

---

### 3. Three experiments to fix the AWS failures

#### Experiment 1 — Upgrade the meaning-detector

The hypothesis: the model converting text to numbers (the embedder) treats "S3" and "GCP Storage" as too similar. A larger, more powerful model should tell them apart.

Result: no change. Both models scored identically. The problem isn't the model's precision — AWS and Google cloud storage really are described with almost the same words across both providers' docs. A smarter model finds that similarity *more* accurately, not less.

| Model | Hit rate | Rank quality |
| --- | --- | --- |
| Standard (22M parameters) | 0.630 | 0.611 |
| Upgraded (110M parameters) | 0.630 | 0.611 |

![Embedding Comparison Chart](eval/embedding_comparison.png)

---

#### Experiment 2 — Add keyword matching on top of meaning search

The hypothesis: if the question contains "S3", results mentioning "S3" by name should rank higher. Old-school keyword counting (BM25 — the same technique behind early search engines) combined with meaning search should recover missed results.

Result: keyword matching made things worse. Hit rate dropped from 0.630 to 0.518. Cloud docs use the same words everywhere — "compute", "storage", "functions", "database" — across all three providers. Counting keywords just scored every provider equally and pushed the right answer down.

| Method | Hit rate | Rank quality |
| --- | --- | --- |
| Meaning search only | 0.630 | 0.611 |
| Keyword + Meaning combined | 0.518 | 0.340 |

![Hybrid Search Chart](eval/hybrid_search_chart.png)

---

#### Experiment 3 — Detect which cloud provider the question is about

The pattern was clear: every failing question was about AWS, and every one retrieved an AWS Lambda result. The hypothesis: scope the search to just that provider's documents.

A classifier was built that detects AWS, Azure, or GCP signals in any question with 100% accuracy across all 27 test questions — including 3 cross-provider questions that correctly returned no filter.

Result: still no improvement. Hit rate stayed at 0.630. But this forced the right question: *if scoping to AWS works perfectly, why does every AWS miss still return Lambda?*

---

### 4. The real problem — the data was broken

Digging into individual failures revealed it: **5 of 6 AWS documentation URLs pointed to broken pages** that loaded almost no content. Amazon S3, EC2, RDS, IAM, and VPC had essentially zero passages stored. AWS Lambda was the only AWS service that had content — so every AWS question retrieved Lambda, regardless of what it was asking.

Three experiments (upgraded embedder, keyword search, provider scoping) all failed not because the approaches were wrong, but because they were searching a broken knowledge base.

**Fix:** replaced all 5 broken URLs with working ones. The knowledge base grew from 50 to 143 passages. A cap of 8 passages per document was added to prevent one large document (S3 had 41 passages) from drowning out smaller ones (Azure Storage had 2). End result: AWS 44 passages, GCP 27, Azure 17.

| State | Hit rate | Rank quality |
| --- | --- | --- |
| Broken knowledge base | 0.630 | 0.611 |
| Fixed knowledge base | 1.000 | 0.975 |
| Fixed + provider scoping | **1.000** | **1.000** |

![Provider Detection Chart](eval/provider_detection_chart.png)

---

### 5. Make the tests honest

Hit rate = 1.000 looked perfect. But every one of the 27 test questions explicitly named the service: "What is Amazon S3?" "What is AWS Lambda?" That's just checking whether the word "S3" appears in the right document — not whether the system can understand what you're asking.

Real users don't ask like that. They say "where do I store customer photos cheaply" or "my function times out connecting to the database." Neither of those questions contains the word "S3" or "VPC."

35 harder questions were added in two rounds:

| Difficulty | Count | Example | What makes it hard |
| --- | --- | --- | --- |
| **Implicit** | 20 | "In AWS, how do I run code without managing servers?" | Cloud provider named, but not the service |
| **Hard** | 15 | "Our database goes down during maintenance — how do we fix it?" | Business language with no cloud terminology at all |

**Why this matters:** a system that scores perfectly on easy questions but fails 8% of real-world questions is failing silently in production — returning a plausible-sounding but wrong answer with full confidence.

**Results across 62 questions:**

| Test set | Hit rate | Rank quality | Grounding |
| --- | --- | --- | --- |
| Easy only (27 questions) | 1.000 | 1.000 | — |
| + Implicit (47 questions) | 1.000 | 0.922 | 4.13 / 5 |
| + Hard (62 questions) | **0.919** | **0.812** | **4.24 / 5** |

5 genuine failures remained — all in security or networking, all asked in plain business English with no technical keywords.

![Progression Chart](eval/progression_chart.png)

![Category Chart](eval/category_chart.png)

**The remaining weakness:** a question like "a junior developer shouldn't be able to delete the production database" has zero IAM or permissions keywords. The system has no way to connect that business description to the AWS IAM documentation.

---

### 6. Smarter retrieval — teaching the system to bridge the vocabulary gap

The problem with the 5 remaining failures: the question uses business language ("shouldn't be able to delete") but the documentation uses technical language ("IAM policies", "least privilege", "role-based access control"). They mean the same thing, but the words don't match.

Three approaches were tested:

**Approach 1 — Generate a hypothetical answer first (HyDE)**
Instead of searching with the original question, ask the AI to first write a short passage that *would* answer it — then search with that. The AI's generated passage naturally contains the technical terms ("IAM policy", "least privilege") because it knows what the answer should look like. That passage retrieves the right document.

**Approach 2 — Second-pass re-ranking**
Retrieve 10 candidate documents with the fast meaning-search, then run a slower, more thorough scorer that reads both the question and each document together to pick the best 3. More accurate — but also slower.

**Approach 3 — HyDE + re-ranking combined**
Use HyDE for the initial retrieval, then re-rank the results.

| Method | Hit rate | Rank quality | Notes |
| --- | --- | --- | --- |
| Baseline | 0.919 | 0.812 | 5 misses |
| Re-ranking only | 0.919 | 0.828 | Fixed 2, broke 1 |
| HyDE + Re-ranking | 0.919 | 0.841 | Fixed 3, broke 2 |
| **HyDE only** | **0.952** | **0.860** | **Fixed 2, broke 0** |

HyDE alone is the winner. Re-ranking hurt when combined with it — because the re-ranker scored documents against the *original* plain-English question, which demoted the technically-worded documents that HyDE had correctly surfaced.

![Improvement Chart](eval/improvement_chart.png)

HyDE pushes the hit rate from 0.919 to 0.952. Three failures remain — all require finding information across multiple documents (e.g., "Lambda times out" → the answer is in the VPC networking docs, not the Lambda docs). That kind of cross-document reasoning is beyond what this system can do in a single retrieval step.

---

## Why this branch exists

At the end of step 6, the system had a hit rate of 0.952. It worked. But "it works" is not the bar for a senior ML engineer role — and that was the target.

The gap between a junior build and a senior one isn't the algorithm. It's the thinking around it. A junior engineer ships when the demo passes. A senior engineer asks three harder questions before calling it done:

1. **Does it fail gracefully when my assumptions break?** — Multi-query reformulation was an attempt to push the hit rate past 0.952 by asking every question multiple ways at once. The honest result was a regression. Reporting that honestly, and diagnosing *why*, is more valuable than only showing the method that scored best.

2. **Do I actually know what the answers look like?** — Hit rate tells you whether the right document was found, not whether the generated answer was correct. Running a proper end-to-end quality check revealed that the system retrieves the right content most of the time, but also pulls in 2 unnecessary passages for every 1 useful one. That's invisible in hit rate alone.

3. **Would this survive real traffic?** — Every retrieval improvement adds response time. Before recommending a method for production, you need to know what each improvement actually costs — in milliseconds and dollars, not just in accuracy points.

The three additions in this branch are not about chasing a higher number. They're about demonstrating the habits that make a system trustworthy enough to deploy and defend in a design review.

---

### 7. Asking the question multiple ways — and an honest regression

The 3 remaining failures all had the same root cause: the question vocabulary doesn't match the document vocabulary, and generating one hypothetical answer still drifts toward the wrong topic. The fix: generate 3 differently-worded versions of the same question, search with all of them, combine the results, and re-rank everything against the original question.

The hypothesis: one of those 3 versions would phrase it as "Lambda VPC subnet security group" — hitting the networking docs that both the original question and HyDE's single pass missed.

**Result: hit rate = 0.903 — a regression from HyDE's 0.952.**

| Method | Hit rate | Rank quality | Notes |
| --- | --- | --- | --- |
| HyDE (best previous) | 0.952 | 0.860 | 3 remaining failures |
| **Multi-question search** | **0.903** | **0.833** | Fixed 1, introduced 2 new failures |

It fixed one failure (the deleted-database scenario) but broke two new ones — questions like "What is Google Cloud's equivalent of an Amazon EC2 instance?" Those need a precise one-to-one match, and generating vocabulary-diverse rewrites actually confused the search.

**The lesson:** more complexity doesn't mean better results. The right question is "what *type* of failure am I trying to fix, and does this technique match that type?" Rewording works for vocabulary mismatch problems. It backfires on precise equivalence questions where the original phrasing was already close enough.

The 3 Lambda/VPC failures are still open. They'd require the system to look across multiple documents and connect information — a fundamentally different capability than retrieval.

---

### 8. Measuring the full picture — not just whether it found the right page

All previous measurements only checked whether the right document was retrieved. They didn't check whether the final answer was actually correct. Three new measurements were added:

- **Retrieval precision** — of the 3 passages shown to the AI, how many were actually needed to answer the question? (1.0 = every passage was useful, 0.0 = all noise)
- **Coverage** — did the retrieved passages contain all the key facts needed to answer correctly?
- **Answer accuracy** — does the AI's generated answer agree with a known reference answer on key facts?

Results on 10 random questions using HyDE retrieval:

| Measurement | Score | What it means |
| --- | --- | --- |
| Retrieval precision | **0.300** | Only 1 of 3 retrieved passages was actually needed |
| Coverage | **0.823** | The right information was present 82% of the time |
| Answer accuracy | **0.520** | Answers were partially correct on average |

The retrieval precision number (0.300) is the most useful finding. The system almost always *has* the right information in what it retrieves — but it comes with 2 extra irrelevant passages. Those don't break the answer, but in a production system they add noise, cost, and latency. The fix would be more targeted filtering — narrowing the search by provider or topic before retrieving.

**One caveat:** the same AI model that generates the answer is also the one judging whether it's correct. A model can't reliably catch its own mistakes. In production, a separate independent judge would be used.

---

### 9. How fast is each method, and what does it cost?

Every improvement to retrieval accuracy adds response time. The system was profiled to measure exactly where time is spent across all three methods:

| Step | Basic search | HyDE | Multi-question |
| --- | --- | --- | --- |
| Convert question to numbers | 17ms | 22ms | 35ms |
| Search the database | 3ms | 2ms | 7ms |
| Generate hypothetical answer | — | 829ms | — |
| Rewrite question 3 ways | — | — | 1,124ms |
| Re-rank results | — | — | 386ms |
| Generate final answer | 124ms | ~800ms* | ~800ms* |
| **Total (typical)** | **~143ms** | **~800–1,200ms** | **~1,500–2,000ms** |

*The raw numbers show 7,000ms for answer generation in HyDE and multi-question — that's because the free API hit its per-minute token limit mid-test. Expected time in production (paid tier) is ~800ms.

**What it costs:** about $0.00003–0.00005 per question. At 10,000 questions per day, total cost is under $0.50/day. Cost isn't the concern — response time is. HyDE adds roughly one extra second per question (the hypothetical generation step). For a live chat interface, that's noticeable.

---

## Techniques used

| What | How |
| --- | --- |
| **Finding relevant passages** | Meaning-based search (converts text to numbers, finds closest match) |
| | Keyword-based search (counts word frequency) — tested, made things worse |
| | Combined keyword + meaning search — tested, also made things worse |
| | HyDE — generate a hypothetical answer first, then search with that |
| | Second-pass re-ranking — slower but more accurate relevance scoring |
| | Multi-question search — ask 3 reworded versions, combine results |
| **Testing the AI model** | MiniLM (fast, small) vs BGE (slower, larger) — no difference in accuracy |
| **Measuring quality** | Hit rate and rank quality (did it find the right doc, how highly ranked?) |
| | AI-as-judge grounding score (does the answer stay within what was retrieved?) |
| | 3-tier test difficulty (easy → implicit → plain business language) |
| | End-to-end quality check (retrieval precision, coverage, answer accuracy) |
| | Failure classification (vocabulary mismatch, topic drift, missing keywords, cross-service) |
| **Tuning** | Swept passage count k=1,3,5,10 — k=3 is the sweet spot |
| | Per-step latency profiling to measure production cost of each technique |
| **Conversation** | Pronoun resolution ("it", "that service") using conversation history |
| | Provider detection to scope searches to the right cloud vendor |

---

## What went wrong

| # | What happened | What it taught |
| --- | --- | --- |
| 1 | Assumed a bigger model would fix retrieval | Model size isn't the problem when the issue is in the data |
| 2 | Assumed keyword search would help | Cloud docs use the same words everywhere — keyword counts score all providers equally |
| 3 | Assumed provider scoping would fix it | When experiments don't move the needle, look at individual failures — not the average |
| 4 | 5 AWS pages returned near-empty content | Three experiments all failed because they were running on broken data, not bad algorithms |
| 5 | One line of code silently deleted the database | A setup function was called twice — second call wiped the data. Always verify after writing. |
| 6 | One document dominated the knowledge base | Amazon S3 had 41 passages vs Azure Storage's 2 — added a cap of 8 per document to balance it |
| 7 | The test set was too easy | Perfect scores on name-matching questions is not a signal. Tests need to reflect how real users actually ask. |
| 8 | Multi-question search was worse than HyDE | More complexity doesn't mean better results. Match the technique to the specific failure mode. |
| 9 | Ran two heavy tests at the same time | Both competed for the same API rate limit — each delayed the other. Run sequentially. |

---

## Branch guide

| Branch | What it covers |
| --- | --- |
| `main` | Complete system with all findings |
| `Evaluation-Set-for-the-RAG` | First test set — 27 questions, baseline measurement |
| `ML-fine-tuning` | Finding the optimal passage count (k-sweep) |
| `ML-fine-tuning-hybrid-search-embedding-model` | Testing a bigger model and keyword search — both failed honestly |
| `automatic-provider-detection` | Provider classifier, data fix, harder test questions |
| `harder-evals-model-improvement` | Expanding to 62 questions — exposes the real 8% failure rate |
| `hyde-rerank-improvement` | HyDE and re-ranking experiments — HyDE wins: hit rate 0.919 → 0.952 |
| `senior-ml-improvements` | Multi-question search, end-to-end quality measurement, latency profiling — production-readiness work |

---

## Time spent

Built across 2 days, ~9 hours total.

| Session | Date | Duration | What was done |
| --- | --- | --- | --- |
| 1 | Apr 9 | 2h 37m | Full pipeline (scraping, search, AI answering, chat), GCP docs, live deployment |
| 2 | Apr 10 | 6h 37m | Test framework, tuning, 3 retrieval experiments, data fix, harder tests, charts, docs |

About 3h 20m of session 2 was waiting for experiments to run. Active coding time: ~5h 54m.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your Groq API key
python step1_ingest.py
python step2_embed_store.py
python step3_rag_query.py
python step4_chat.py
```
