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

Recall plateaus completely at K=3. Going to K=5 or K=10 adds zero retrieval gain while faithfulness actually drops — the extra chunks introduce irrelevant context that dilutes the answer. K=3 is now the system default.

![K-Sweep Chart](eval/k_sweep_chart.png)

---

## Embedding model comparison

To test whether a stronger embedding model would improve recall, `BAAI/bge-base-en-v1.5` (110M params, 768-dim, trained specifically for retrieval tasks) was benchmarked against the current `all-MiniLM-L6-v2` (22M params, 384-dim, general purpose).

| Model | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| MiniLM (current) | 0.630 | 0.611 | 0.630 | 0.611 |
| BGE (new) | 0.630 | 0.611 | 0.630 | 0.611 |

Both models score identically. This is an important finding — the bottleneck is not the embedding model. S3 and GCP Storage are genuinely semantically similar concepts, and no dense embedding model will distinguish them without additional signal. The real fix is provider-aware metadata filtering at query time or hybrid search (BM25) which matches on exact service names like "S3" or "IAM" regardless of semantic similarity.

![Embedding Comparison Chart](eval/embedding_comparison.png)

---

## Performance improvements

The first version worked but was slow. A few targeted changes made a significant difference:

- **Parallel fetching** — 18 docs fetched one at a time with a 1s sleep between each. `ThreadPoolExecutor` eliminated ~18s of dead wait.
- **Batch embeddings** — per-chunk `model.encode()` loop replaced with a single `model.encode(all_chunks, batch_size=64)` call.
- **Cached singletons** — ChromaDB and tiktoken were re-initialized on every query. Module-level singletons removed that overhead entirely.
- **Real streaming** — original "streaming" was fake: wait for full response, print word-by-word with `sleep(0.05)`. The actual Groq streaming API delivers the first token in ~100-300ms.

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
