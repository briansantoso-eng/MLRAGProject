# CloudDocs RAG System

**[Try the live demo](https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/)**

---

Cloud documentation is scattered across three providers, each with its own terminology and structure. Searching it manually is slow. Asking a general-purpose LLM is fast but unreliable — it hallucinates service names, outdated pricing, and non-existent features.

This project solves that by building a RAG system grounded in the actual docs. It scrapes real AWS, Azure, and GCP documentation, stores it in a vector database, and retrieves the most relevant passages before generating an answer — so every response is traceable back to a source.

![RAG Chat Screenshot](ss/ss1.png)

---

## NLP Techniques

| Category | Technique |
| --- | --- |
| **Retrieval** | Dense vector search (cosine similarity) |
| | BM25 keyword retrieval (term frequency scoring) |
| | Hybrid search with Reciprocal Rank Fusion (RRF) |
| **Embedding models** | Bi-encoder comparison: MiniLM vs BGE (model size vs retrieval quality) |
| **Evaluation** | Recall@K — retrieval coverage |
| | MRR@K — ranking quality |
| | LLM-as-judge — automated answer quality scoring |
| | Eval set hardening — 3 tiers of difficulty (explicit → implicit → hard) to surface realistic failure modes |
| **Hyperparameter tuning** | K-sweep — systematic search over retrieval depth (k=1,3,5,10) |
| | Recall-faithfulness tradeoff analysis |
| **Query understanding** | Query rewriting — pronoun resolution using conversation history |
| | Provider disambiguation — diagnosing that retrieval failures stem from missing query intent, not model quality |

**Key insight from experiments:** when two retrieval approaches both fail, the problem is usually in the data structure or query understanding — not the algorithm. Dense search and BM25 both struggled for the same reason: cloud docs from different providers describe the same concepts using identical words. No retrieval algorithm can distinguish them without knowing which provider the user is asking about.

---

## Branch guide

Each branch in this repo represents a distinct stage of the ML development process:

| Branch | What it covers |
| --- | --- |
| `main` | Complete system with all findings merged in |
| `Evaluation-Set-for-the-RAG` | Building the 27-question ground-truth eval set and measuring baseline performance at k=5 |
| `ML-fine-tuning` | K-sweep across k=1,3,5,10 to find the optimal number of retrieved chunks |
| `ML-fine-tuning-hybrid-search-embedding-model` | Two retrieval improvement experiments: BGE embedding model swap and BM25 hybrid search — both tested against the eval set with honest results |
| `automatic-provider-detection` | Built a keyword-based provider classifier (100% accurate on eval set) and integrated it into the pipeline — reveals the true root cause is corpus imbalance, not cross-provider confusion |

---

## How it works

Real cloud docs are fetched, chunked into overlapping passages, and embedded locally using SentenceTransformers. At query time, the question is embedded the same way, and ChromaDB retrieves the closest matching passages. Those passages — not the model's prior knowledge — form the basis of the answer, generated via Groq Llama 3 8B.

Groq is used for its free tier. GPT-4o would be the production choice with funding.

**ML techniques used:**

| Layer | Tool / Technique |
| --- | --- |
| Text segmentation | Sliding window chunking (1000 chars, 200 overlap) with sentence boundary detection |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` — 384-dim dense vectors, local and free |
| Vector search | ChromaDB with HNSW indexing, cosine similarity |
| LLM inference | Groq Llama 3.1 8B Instant — low temperature (0.1) for factual answers. Groq used for its free tier; GPT-4o would be the production choice with funding. |
| Query rewriting | Pronoun resolution using conversation history before retrieval |
| Retrieval eval | Recall@K, MRR@K (Mean Reciprocal Rank) |
| Answer eval | LLM-as-judge faithfulness scoring (1-5) |
| Hyperparameter tuning | K-sweep across k=1,3,5,10 to find optimal retrieval size |
| Throughput | Parallel fetching (ThreadPoolExecutor), batch encoding (batch_size=64), singleton caching |

---

## Measuring whether it actually works

Saying "it seems to work" isn't enough. A ground-truth eval set was built across all three providers and six categories (compute, storage, database, security, networking, cross-provider). Each question has a known expected source, so retrieval quality can be scored automatically.

Three metrics:

- **Recall@K** — did the right document appear in the top K results?
- **MRR@K** — how highly was it ranked?
- **Faithfulness** — does the answer stay grounded in what was retrieved? Scored 1-5 by a second LLM call acting as judge.

The chart below shows every tuning experiment and every eval difficulty tier in a single view, ordered from the starting point to the final honest numbers:

![Progression Chart](eval/progression_chart.png)

### Eval set: three tiers of difficulty

The eval set was built in three phases, each harder than the last:

| Tier | Count | Example | What it tests |
| --- | --- | --- | --- |
| **Explicit** | 27 q | "What is Amazon S3 and what kind of data can you store in it?" | Name-match retrieval — baseline only |
| **Implicit** | 20 q | "In AWS, how do I run event-driven code without managing servers?" | Semantic retrieval: provider named, service not |
| **Hard** | 15 q | "My Lambda function times out connecting to RDS — what is misconfigured?" | Business problems, cross-vocabulary, failure scenarios, no cloud terms |

The hard tier includes:

- **Business language** — no cloud terminology at all ("our database goes down during maintenance")
- **Cross-vocabulary** — uses one provider's name to ask about another ("what is GCP's equivalent of EC2?")
- **Failure scenarios** — describes a symptom, not a service ("Lambda times out connecting to RDS")
- **Least-privilege security** — asks about access restrictions without naming IAM

Final eval set: **62 questions** across all 6 categories.

### Final results (fixed corpus + 62-question eval set)

| Metric | Easy (27 q) | Implicit (47 q) | Hard (62 q) |
| --- | --- | --- | --- |
| Recall@3 | 1.000 | 1.000 | **0.919** |
| MRR@3 | 1.000 | 0.922 | **0.812** |
| Faithfulness | — | 4.13 / 5 | **4.24 / 5** |

Recall dropped to 0.919 — 5 questions genuinely failed. MRR dropped to 0.812. Faithfulness stayed high at 4.24/5, meaning retrieved context produces grounded answers even on harder queries.

**The 5 misses and why:**

| Question | Expected | Why retrieval failed |
| --- | --- | --- |
| "Developer can deploy code but not drop the database schema" | IAM | No IAM keywords — purely describes a permissions design pattern |
| "Web servers internet-facing, DB servers fully isolated" | VPC | Problem framing with no VPC/subnet terminology |
| "What is the Azure equivalent of AWS IAM?" | Azure Active Directory | "plays the same role as" — indirect phrasing without identity keywords |
| "Lambda times out connecting to RDS — what is misconfigured?" | Amazon VPC | Symptom-based — correct answer (Lambda needs VPC config) requires multi-hop reasoning |
| "Junior developer deleted the DB — what access control was missing?" | IAM | Incident description, zero IAM terminology |

All 5 misses fall in security or networking. Both categories use conceptual language — "least privilege", "isolation", "access control" — that the embedding model struggles to map to specific service chunks.

![Category Chart](eval/category_chart.png)

The heatmap makes the weakness immediately visible: everything is 100% through Explicit and Implicit tiers, then Networking drops to 0% and Security to 25% on Hard questions. The faithfulness bars confirm the same pattern — when retrieval fails, it tends to fail on these two categories.

| Category | Recall@3 | Faithfulness |
| --- | --- | --- |
| Compute | 100% (15/15) | 4.4 / 5 |
| Storage | 100% (10/10) | 4.6 / 5 |
| Database | 100% (8/8) | 4.4 / 5 |
| Networking | 78% (7/9) | 4.3 / 5 |
| Cross-provider | 100% (9/9) | 3.8 / 5 |
| Security | 73% (8/11) | 3.9 / 5 |

### Why harder eval questions matter

An eval set that's too easy doesn't tell you anything useful — it just confirms the plumbing works. The three difficulty tiers were designed to surface increasingly realistic failure modes:

**Explicit questions** confirm basic retrieval: if "What is Amazon S3?" doesn't return the S3 doc, nothing is working. Recall=1.000 here is expected, not impressive. It's a sanity check.

**Implicit questions** test whether the system does actual semantic retrieval or just name-matching. "In AWS, how do I host a static website from object storage?" contains the provider signal but not the service name — the system has to understand that object storage + static hosting = S3. Getting these right means the embeddings are capturing meaning, not just keywords.

**Hard questions** simulate real users. Real users don't say "AWS IAM" — they say "a developer shouldn't be able to delete the production database." They don't say "Amazon VPC" — they describe the symptom: "my Lambda function times out connecting to RDS." These questions found 5 genuine failures, all in security and networking, that the easier tiers completely missed.

**The production benefit:** a system that scores 1.000 on explicit questions but 0.919 on hard questions is telling you that 8% of real-world queries will fail silently — returning a plausible but wrong document, generating a confident but wrong answer. Knowing that in advance is what makes the difference between a demo and something you'd trust in production.

---

### Baseline (before corpus fix)

These numbers are from the original broken corpus, included here so the progression is traceable:

| Metric | Score |
| --- | --- |
| Recall@5 | 0.63 (17/27 hits) |
| MRR@5 | 0.61 |
| Faithfulness | 3.7 / 5 |

AWS-specific questions missed consistently because 5 of 6 AWS docs were broken SPA pages returning 21 characters. Every AWS query retrieved Lambda because it was the only AWS document that existed.

---

## Fine-tuning summary

Every tuning decision made during development, in order:

| # | What was tuned | Why | Outcome |
| --- | --- | --- | --- |
| 1 | K-sweep (k=1,3,5,10) | Didn't know the right number of retrieved chunks — too few misses information, too many adds noise | k=3 is optimal: recall plateaus there, faithfulness peaks, cost is lowest |
| 2 | BGE embedding model swap | Believed a larger, retrieval-specific model would produce more discriminative embeddings | No change — cloud services are semantically equivalent across providers; no model can separate them by meaning alone |
| 3 | BM25 hybrid search | Expected keyword frequency matching to recover exact service name matches that dense search missed | Hurt recall at k=3 (0.630 → 0.518) — cloud docs use identical vocabulary so BM25 scores every provider equally |
| 4 | Automatic provider detection | Spotted pattern that all 10 misses were AWS queries retrieving Lambda — suspected cross-provider confusion | 100% detection accuracy (27/27) but recall unchanged — forced inspection of the real root cause |
| 5 | Corpus URL fix | Detection experiment revealed 5 AWS URLs pointed to JS SPA pages returning 21 chars, not actual docs | Recall jumped 0.630 → 1.000 after re-scraping with working URLs (143 chunks vs 50) |
| 6 | MAX_CHUNKS_PER_DOC cap | After URL fix, S3 had 41 chunks vs Azure Storage's 2 — imbalance would bias retrieval toward AWS | Balanced corpus: AWS 44, GCP 27, Azure 17 chunks |
| 7 | Implicit eval questions (+20) | All 27 original questions named the service directly — Recall=1.000 was trivially achieved by name-matching, not semantics | MRR dropped 1.000 → 0.922 — harder ranking challenge with no recall loss |
| 8 | Hard eval questions (+15) | Implicit questions still had provider signals; real users describe problems without cloud terminology | Recall dropped to 0.919, MRR to 0.812 — first genuine retrieval failures, in security and networking |

---

## Finding the optimal K

**Why this was done:** K is the single most important retrieval hyperparameter — it controls how much context the LLM sees. There was no principled reason to start at k=5, so a sweep was run to find the actual optimum rather than guessing.

K controls how many retrieved passages get passed to the LLM. Too few and you miss relevant information. Too many and you flood the context with noise — and pay more per query.

A sweep across K = 1, 3, 5, 10 found the answer quickly:

| K | Recall | Faithfulness | Time |
| --- | --- | --- | --- |
| 1 | 0.593 | 3.85 | 133s |
| **3** | **0.630** | **4.11** | **308s** |
| 5 | 0.630 | 3.74 | 399s |
| 10 | 0.630 | 3.93 | 641s |

Recall plateaus completely at K=3. Going higher adds zero retrieval gain while faithfulness drops — extra chunks introduce irrelevant context that dilutes the answer. K=3 is now the system default.

![K-Sweep Chart](eval/k_sweep_chart.png)

---

## Retrieval improvement experiments

Recall was stuck at 0.630 after the K-sweep. Three more experiments were run in sequence, each motivated by a hypothesis about why retrieval was failing.

### Experiment 1: Stronger embedding model (BGE)

**Why:** The working theory was that `all-MiniLM-L6-v2` (22M parameters, general-purpose) was encoding cloud service descriptions too loosely — S3 and GCP Storage landing close together in vector space because the model wasn't trained specifically for retrieval. A 5x larger model trained on retrieval tasks should produce tighter, more discriminative embeddings.

Hypothesis: `all-MiniLM-L6-v2` is a small general-purpose model. Swapping to `BAAI/bge-base-en-v1.5` (5x larger, trained specifically for retrieval) should produce more discriminative embeddings.

**Result: no improvement.** Both models scored identically.

| Model | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| MiniLM (384-dim, 22M params) | 0.630 | 0.611 | 0.630 | 0.611 |
| BGE (768-dim, 110M params) | 0.630 | 0.611 | 0.630 | 0.611 |

The bottleneck is not model quality. S3 and GCP Storage genuinely are semantically similar — they are both object storage services. No embedding model will separate them because their meanings are nearly identical.

![Embedding Comparison Chart](eval/embedding_comparison.png)

### Experiment 2: Hybrid search (BM25 + Dense via RRF)

**Why:** If the embedding model was not the bottleneck, maybe the retrieval method was. BM25 ignores meaning entirely — it scores based on raw token frequency. A query containing "S3" should boost Amazon S3 above GCP Storage purely because the token "S3" appears more in S3 docs. Combining it with dense retrieval via Reciprocal Rank Fusion (RRF) should get the best of both worlds: semantic match from dense vectors and keyword precision from BM25.

Hypothesis: BM25 scores on exact token frequency, not semantics. A query containing "S3" should give "Amazon S3" a high BM25 score regardless of how similar GCP Storage is in meaning. Combining BM25 with dense retrieval via Reciprocal Rank Fusion should recover keyword matches that dense search misses.

**Result: BM25 made things worse.**

| Method | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| Dense only | 0.630 | 0.611 | 0.630 | 0.611 |
| Hybrid BM25 + Dense | 0.518 | 0.340 | 0.630 | 0.365 |

At k=3 recall dropped from 0.630 to 0.518. At k=5 recall matched but MRR fell — correct docs were ranked lower. The reason: every AWS, Azure, and GCP doc contains the same words ("storage", "compute", "functions", "database"). BM25 scores all of them equally high for any query, adding noise that pushes correct results down.

![Hybrid Search Chart](eval/hybrid_search_chart.png)

### Experiment 3: Automatic provider detection

**Why:** After two failed algorithm experiments, the miss pattern was inspected directly instead of trying another algorithm. Every one of the 10 failing questions was an AWS query, and every one of them retrieved Lambda as the top result. The hypothesis shifted: this wasn't a retrieval algorithm problem — it was a query scope problem. The system had no way to know a user asking about S3 didn't also want GCP Storage results. Adding provider detection to scope retrieval would eliminate cross-provider noise without changing the algorithm at all.

After the first two experiments failed, inspecting the actual misses revealed a pattern: all 10 failing questions were AWS queries, and every one of them retrieved "AWS Lambda" as the top result instead of the correct service. This suggested the problem was cross-provider confusion — Lambda vs GCP Storage, for example.

A keyword-based provider classifier was built (`provider_detector.py`) and integrated into the pipeline. It detects provider signals in the query ("S3" → aws, "Azure Blob" → azure) and scopes ChromaDB retrieval to that provider, eliminating cross-provider noise entirely.

**Detection accuracy: 27/27 correct (100%).** Every question — including the 3 cross-provider queries which correctly returned no filter.

**Result: no improvement.** Recall stayed at 0.630.

| Method | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| Baseline (no filter) | 0.630 | 0.611 | 0.630 | 0.611 |
| Auto provider detection | 0.630 | 0.611 | 0.630 | 0.611 |

![Provider Detection Chart](eval/provider_detection_chart.png)

### What all three experiments revealed

After experiment 3, inspecting the miss pattern revealed the real root cause: **5 out of 6 AWS documentation URLs were broken**. The original URLs pointed to JavaScript-rendered index pages (`docs.aws.amazon.com/s3/index.html`) that return only 21 characters of content. Amazon S3, EC2, RDS, IAM, and VPC had 0 chunks in the index — only Lambda was successfully scraped.

Every AWS query retrieved Lambda because it was the only AWS document that existed. All previous experiments (BGE, BM25, provider detection) showed no improvement because they were running against a broken corpus.

**Fix:** replaced all 5 broken URLs with direct content page URLs pointing to the actual documentation. After re-scraping (143 chunks vs 50 before) and re-embedding:

| Method | Recall@3 | MRR@3 |
| --- | --- | --- |
| Baseline — broken corpus (27 easy q) | 0.630 | 0.611 |
| Baseline — fixed corpus (27 easy q) | 1.000 | 0.975 |
| Auto provider detection — fixed corpus (27 easy q) | 1.000 | 1.000 |
| Fixed corpus + 47 questions (20 implicit) | 1.000 | 0.922 |
| **Fixed corpus + 62 questions (35 implicit + hard)** | **0.919** | **0.812** |

Recall dropped to 0.919 and MRR to 0.812 after adding the hardest questions — first time recall fell below 1.000. These are the honest production numbers.

**Final corpus after balancing:** AWS 44, GCP 27, Azure 17 — capped at 8 chunks per document via `MAX_CHUNKS_PER_DOC` in config.

![Provider Detection Chart](eval/provider_detection_chart.png)

---

## Mistakes made and what they revealed

A full log of every error made during development — what was wrong, why it happened, and what it taught.

| # | Mistake | What actually happened | Fix |
| --- | --- | --- | --- |
| 1 | Predicted BGE would improve recall | BGE scored identically to MiniLM (Recall=0.630 both). The problem was not model size — cloud services are semantically equivalent across providers by definition | Ran the experiment instead of assuming; confirmed the bottleneck was elsewhere |
| 2 | Predicted BM25 hybrid search would improve recall | BM25 made recall worse at k=3 (0.518 vs 0.630). Cloud docs use identical vocabulary across providers so BM25 scores them all equally, adding noise | Measured the result honestly; ruled out keyword frequency as the fix |
| 3 | Predicted provider metadata filtering would improve recall | Detection was 100% accurate but recall didn't move. All 10 misses were AWS queries returning Lambda — even within AWS-only results | Forced inspection of individual misses, which revealed the real problem |
| 4 | AWS documentation URLs pointed to index pages | `docs.aws.amazon.com/s3/index.html` returns 21 chars (a JavaScript SPA shell). S3, EC2, RDS, IAM, VPC all had 0 chunks. Every AWS query retrieved Lambda because it was the only AWS doc that existed | Replaced all 5 broken URLs with direct content page URLs; corpus grew from 50 to 143 chunks |
| 5 | `step2_embed_store.py` called `initialize_chroma()` twice | `main()` ran `process_documents()` (builds collection), then immediately called `initialize_chroma()` again which deleted and recreated the collection — wiping all data silently. Collection always showed 0 chunks after embed | Made `process_documents()` return the collection; `main()` reuses it instead of reinitializing |
| 6 | Corpus imbalance after URL fix | S3 returned 41 chunks, RDS 25 — while Azure Storage had 2. AWS dominated 99/143 chunks, which would bias retrieval for ambiguous real-world queries | Added `MAX_CHUNKS_PER_DOC = 8` cap in config; corpus balanced to AWS 44, GCP 27, Azure 17 |
| 7 | Eval set was too easy | All 27 original questions explicitly named the service ("What is Amazon S3..."). Recall=1.000 on this set is trivially achieved by name-matching — it doesn't test real retrieval quality | Added 35 harder questions: implicit use-case descriptions, business language, failure scenarios, cross-vocabulary queries. Recall dropped to 0.919 and MRR to 0.812 — the honest numbers |

**The core lesson:** three retrieval algorithm experiments (BGE, BM25, provider detection) all failed because the data was broken, not the algorithms. Debugging retrieval quality problems by inspecting individual misses — not just aggregate metrics — is what revealed the root cause. The eval set lesson is the same: if your test questions are easy, a perfect score tells you nothing.

---

## Performance improvements

- **Parallel fetching** — 18 docs fetched one at a time with a 1s sleep between each. `ThreadPoolExecutor` eliminated ~18s of dead wait.
- **Batch embeddings** — per-chunk `model.encode()` loop replaced with `model.encode(all_chunks, batch_size=64)`.
- **Cached singletons** — ChromaDB and tiktoken re-initialized on every query. Module-level singletons removed that overhead entirely.
- **Real streaming** — original "streaming" was fake: wait for full response, print word-by-word with `sleep(0.05)`. Groq streaming API delivers first token in ~100-300ms.

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
