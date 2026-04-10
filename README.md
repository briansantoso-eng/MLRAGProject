# CloudDocs RAG System

A retrieval-augmented generation system built on real AWS, Azure, and GCP documentation. Ask questions, get grounded answers.

## Live Demo

**[https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/](https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/)**

## Stack

- **LLM:** Groq Llama 3 8B
- **Embeddings:** SentenceTransformers all-MiniLM-L6-v2 (local, free)
- **Vector DB:** ChromaDB with HNSW indexing

## How it works

1. Scrapes real cloud documentation (AWS, Azure, GCP)
2. Chunks and embeds text into ChromaDB
3. Retrieves relevant passages on query
4. Generates grounded answers via Groq

## Evaluation Framework

A 27-question ground-truth eval set (`eval_dataset.json`) covers 18 source documents across AWS, Azure, and GCP plus 3 cross-provider comparisons. Each question maps to a known expected source, enabling automated scoring without human review.

Three metrics measured via `step5_evaluate.py`:

- **Recall@K** — did the right document appear in the top-K retrieved chunks?
- **MRR@K** — how highly ranked was it? (rewards rank 1 over rank 5)
- **Faithfulness** — does the answer stay grounded in retrieved context? Scored 1–5 by a second LLM-as-judge call.

**Baseline results (k=5):**

| Metric | Score |
| --- | --- |
| Recall@5 | 0.63 (17/27 hits) |
| MRR@5 | 0.61 |
| Avg Faithfulness | 3.7 / 5 |

| Category | Recall@5 | Faithfulness |
| --- | --- | --- |
| Compute | 75% | 4.0 / 5 |
| Cross-provider | 100% | 4.3 / 5 |
| Storage | 50% | 2.8 / 5 |
| Database | 50% | 2.8 / 5 |
| Security | 50% | 4.0 / 5 |
| Networking | 50% | 4.3 / 5 |

**Key finding:** Cross-provider queries hit 100% recall because either cloud's doc satisfies the match. AWS-specific queries frequently retrieve semantically equivalent Azure or GCP docs instead — `all-MiniLM-L6-v2` treats equivalent services across clouds as near-identical vectors (S3 ≈ GCP Storage ≈ Azure Blob Storage).

## Hyperparameter Tuning (K-Sweep)

K controls how many retrieved chunks are passed to the LLM — directly trading off recall, answer quality, cost, and latency. To find the optimal value, a sweep was run at K = 1, 3, 5, 10 with and without faithfulness scoring.

| K | Faithfulness | Recall@K | MRR@K | Avg Faith | Time (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | No | 0.593 | 0.593 | — | 1.9 |
| 1 | Yes | 0.593 | 0.593 | 3.85 | 133.4 |
| 3 | No | 0.630 | 0.611 | — | 0.5 |
| **3** | **Yes** | **0.630** | **0.611** | **4.11** | **308.5** |
| 5 | No | 0.630 | 0.611 | — | 0.9 |
| 5 | Yes | 0.630 | 0.611 | 3.74 | 398.7 |
| 10 | No | 0.630 | 0.611 | — | 0.6 |
| 10 | Yes | 0.630 | 0.611 | 3.93 | 641.2 |

**Result: K=3 is optimal.** Recall plateaus completely at K=3 — K=5 and K=10 add zero improvement. Faithfulness peaks at K=3 (4.11/5) and drops at K=5, because extra chunks introduce irrelevant context that dilutes the answer. K=3 was set as the system default.

![K-Sweep Chart](eval/k_sweep_chart.png)

Run the eval yourself:

```bash
python step5_evaluate.py --sweep            # full k-sweep (k=1,3,5,10)
python step5_evaluate.py --k 3              # single run at optimal K
python step5_evaluate.py --no-faithfulness  # retrieval only, no API cost
```

## Performance Optimizations

| File | Change | Impact |
| --- | --- | --- |
| `step1_ingest.py` | Sequential fetch + `sleep(1)` → `ThreadPoolExecutor(max_workers=5)` | 18 docs fetched concurrently; ~18s of dead wait eliminated |
| `step2_embed_store.py` | Per-chunk `model.encode()` loop → `model.encode(all_chunks, batch_size=64)` | Batch mode is significantly faster; single ChromaDB `add()` per doc |
| `step3_rag_query.py` | New ChromaDB client + tiktoken encoder on every query → module-level singletons | Avoids re-initializing disk-backed DB and tokenizer per call |
| `step4_chat.py` | Fake streaming (full response + word-by-word sleep) → real Groq streaming API | First token in ~100–300ms instead of waiting for full response |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your Groq API key
python step1_ingest.py
python step2_embed_store.py
python step3_rag_query.py
python step4_chat.py
```

## Screenshot

![RAG Chat Screenshot](ss/ss1.png)
