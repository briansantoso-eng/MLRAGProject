# CloudDocs RAG System

**[Try the live demo](https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/)**

---

Cloud documentation is scattered across three providers, each with its own terminology and structure. Searching it manually is slow. Asking a general-purpose LLM is fast but unreliable — it hallucinates service names, outdated pricing, and non-existent features.

This project solves that by building a RAG system grounded in the actual docs. It scrapes real AWS, Azure, and GCP documentation, stores it in a vector database, and retrieves the most relevant passages before generating an answer — so every response is traceable back to a source.

![RAG Chat Screenshot](ss/ss1.png)

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
| LLM inference | Groq Llama 3.1 8B Instant — low temperature (0.1) for factual answers |
| Query rewriting | Pronoun resolution using conversation history before retrieval |
| Retrieval eval | Recall@K, MRR@K (Mean Reciprocal Rank) |
| Answer eval | LLM-as-judge faithfulness scoring (1-5) |
| Hyperparameter tuning | K-sweep across k=1,3,5,10 to find optimal retrieval size |
| Throughput | Parallel fetching (ThreadPoolExecutor), batch encoding (batch_size=64), singleton caching |

---

## Measuring whether it actually works

A 27-question ground-truth eval set was built across all three providers and six categories. Each question maps to a known expected source, enabling automated scoring.

Three metrics:

- **Recall@K** — did the right document appear in the top K results?
- **MRR@K** — how highly was it ranked?
- **Faithfulness** — does the answer stay grounded in what was retrieved? Scored 1-5 by a second LLM call acting as judge.

Baseline at K=5:

| Metric | Score |
| --- | --- |
| Recall@5 | 0.63 (17/27 hits) |
| MRR@5 | 0.61 |
| Faithfulness | 3.7 / 5 |

| Category | Recall@5 | Faithfulness |
| --- | --- | --- |
| Compute | 75% | 4.0 / 5 |
| Cross-provider | 100% | 4.3 / 5 |
| Storage | 50% | 2.8 / 5 |
| Database | 50% | 2.8 / 5 |
| Security | 50% | 4.0 / 5 |
| Networking | 50% | 4.3 / 5 |

Cross-provider questions hit 100% recall. AWS-specific questions miss often — the embedding model treats S3, GCP Storage, and Azure Blob as near-identical vectors because they are semantically equivalent services.

---

## Finding the optimal K

A sweep across K = 1, 3, 5, 10 showed recall plateaus at K=3 with the highest faithfulness. K=3 is now the system default.

| K | Recall | Faithfulness | Time |
| --- | --- | --- | --- |
| 1 | 0.593 | 3.85 | 133s |
| **3** | **0.630** | **4.11** | **308s** |
| 5 | 0.630 | 3.74 | 399s |
| 10 | 0.630 | 3.93 | 641s |

![K-Sweep Chart](eval/k_sweep_chart.png)

---

## Retrieval improvement experiments

Two approaches were tested to improve Recall@K beyond 0.630.

### Experiment 1: Stronger embedding model (BGE)

Hypothesis: `all-MiniLM-L6-v2` is a small general-purpose model. Swapping to `BAAI/bge-base-en-v1.5` (5x larger, trained specifically for retrieval) should produce more discriminative embeddings.

**Result: no improvement.** Both models scored identically.

| Model | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| MiniLM (384-dim, 22M params) | 0.630 | 0.611 | 0.630 | 0.611 |
| BGE (768-dim, 110M params) | 0.630 | 0.611 | 0.630 | 0.611 |

The bottleneck is not model quality — S3 and GCP Storage genuinely are semantically similar. No embedding model will separate them because their meanings are nearly identical.

![Embedding Comparison Chart](eval/embedding_comparison.png)

### Experiment 2: Hybrid search (BM25 + Dense via RRF)

Hypothesis: BM25 scores on exact token frequency, not semantics. A query containing "S3" should give "Amazon S3" a high BM25 score regardless of semantic similarity to GCP Storage. Combining BM25 with dense retrieval via Reciprocal Rank Fusion should recover the keyword matches that dense search misses.

**Result: BM25 made things worse.**

| Method | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| Dense only | 0.630 | 0.611 | 0.630 | 0.611 |
| Hybrid BM25 + Dense | 0.518 | 0.340 | 0.630 | 0.365 |

At k=3, hybrid recall dropped from 0.630 to 0.518. At k=5, recall matched but MRR fell sharply — meaning correct documents were ranked lower.

The reason: cloud documentation uses identical terminology across all providers. Every AWS, Azure, and GCP doc contains the words "storage", "compute", "virtual machine", "functions", "database", "security". BM25 scores all of them equally high for any query, creating noise that pushes correct results down in the merged ranking.

![Hybrid Search Chart](eval/hybrid_search_chart.png)

### What the experiments revealed

Both approaches failed for the same underlying reason: the retrieval problem here is a **provider disambiguation** problem, not a retrieval quality problem. The correct document exists in the index — the system just can't tell which provider the user is asking about.

The right fix is **automatic provider detection**: classify the query to identify which provider is intended ("S3" → AWS, "Blob Storage" → Azure, "Cloud Functions" → could be GCP or Azure), then apply a metadata filter to scope ChromaDB retrieval to that provider. This is already partially supported in the pipeline via `provider_filter` — making it automatic is the logical next step.

---

## Performance improvements

- **Parallel fetching** — `ThreadPoolExecutor` eliminated ~18s of sequential wait
- **Batch embeddings** — single `model.encode(all_chunks, batch_size=64)` replaces per-chunk loop
- **Cached singletons** — ChromaDB and tiktoken initialized once, not on every query
- **Real streaming** — Groq streaming API delivers first token in ~100-300ms

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
