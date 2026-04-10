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
| | Ground-truth eval set construction (27 labeled question/source pairs) |
| **Hyperparameter tuning** | K-sweep — systematic search over retrieval depth |
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

Saying "it seems to work" isn't enough. A 27-question ground-truth eval set was built across all three providers and six categories (compute, storage, database, security, networking, cross-provider). Each question has a known expected source, so retrieval quality can be scored automatically.

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

Cross-provider questions ("compare Lambda vs Azure Functions") hit 100% recall — either doc satisfies the match. AWS-specific questions miss more often because the embedding model treats S3, GCP Storage, and Azure Blob as near-identical vectors. The model doesn't know they belong to different providers, only that they're semantically similar.

---

## Finding the optimal K

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

Two approaches were tested to push Recall@K beyond 0.630.

### Experiment 1: Stronger embedding model (BGE)

Hypothesis: `all-MiniLM-L6-v2` is a small general-purpose model. Swapping to `BAAI/bge-base-en-v1.5` (5x larger, trained specifically for retrieval) should produce more discriminative embeddings.

**Result: no improvement.** Both models scored identically.

| Model | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| MiniLM (384-dim, 22M params) | 0.630 | 0.611 | 0.630 | 0.611 |
| BGE (768-dim, 110M params) | 0.630 | 0.611 | 0.630 | 0.611 |

The bottleneck is not model quality. S3 and GCP Storage genuinely are semantically similar — they are both object storage services. No embedding model will separate them because their meanings are nearly identical.

![Embedding Comparison Chart](eval/embedding_comparison.png)

### Experiment 2: Hybrid search (BM25 + Dense via RRF)

Hypothesis: BM25 scores on exact token frequency, not semantics. A query containing "S3" should give "Amazon S3" a high BM25 score regardless of how similar GCP Storage is in meaning. Combining BM25 with dense retrieval via Reciprocal Rank Fusion should recover keyword matches that dense search misses.

**Result: BM25 made things worse.**

| Method | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| Dense only | 0.630 | 0.611 | 0.630 | 0.611 |
| Hybrid BM25 + Dense | 0.518 | 0.340 | 0.630 | 0.365 |

At k=3 recall dropped from 0.630 to 0.518. At k=5 recall matched but MRR fell — correct docs were ranked lower. The reason: every AWS, Azure, and GCP doc contains the same words ("storage", "compute", "functions", "database"). BM25 scores all of them equally high for any query, adding noise that pushes correct results down.

![Hybrid Search Chart](eval/hybrid_search_chart.png)

### What the experiments revealed

Both approaches failed for the same reason: this is a **provider disambiguation** problem, not a retrieval quality problem. The correct document exists in the index — the system just cannot tell which provider the user is asking about.

The right fix is **automatic provider detection**: classify the query to identify the intended provider ("S3" → AWS, "Blob Storage" → Azure), then scope ChromaDB retrieval with a metadata filter. This is already partially supported via `provider_filter` — making it automatic is the logical next step.

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
