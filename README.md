# CloudDocs RAG System

Retrieval-augmented generation over real AWS, Azure, and GCP documentation.

## Live Demo

**[https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/](https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/)**

## Stack

- **LLM:** Groq Llama 3 8B
- **Embeddings:** SentenceTransformers all-MiniLM-L6-v2 (local, free)
- **Vector DB:** ChromaDB

## How it works

1. Scrapes real cloud docs (AWS, Azure, GCP)
2. Chunks, embeds, and stores in ChromaDB
3. Retrieves top-K passages on query
4. Generates grounded answers via Groq

## Evaluation

Built a 27-question ground-truth eval set across all three providers and six categories. Metrics: **Recall@K**, **MRR@K**, and **LLM-as-judge Faithfulness** (1–5).

| Metric | k=5 baseline |
| --- | --- |
| Recall@5 | 0.63 |
| MRR@5 | 0.61 |
| Faithfulness | 3.7 / 5 |

Cross-provider queries hit 100% recall. AWS-specific queries frequently miss — the embedding model treats S3, GCP Storage, and Azure Blob as near-identical vectors.

## Hyperparameter Tuning

Swept K = 1, 3, 5, 10 to find the optimal number of retrieved chunks.

| K | Recall | Faithfulness | Time (s) |
| --- | --- | --- | --- |
| 1 | 0.593 | 3.85 | 133s |
| **3** | **0.630** | **4.11** | **308s** |
| 5 | 0.630 | 3.74 | 399s |
| 10 | 0.630 | 3.93 | 641s |

**K=3 is optimal.** Recall plateaus at K=3 with zero gain at K=5 or K=10. Faithfulness peaks at K=3 — more chunks dilute the answer with irrelevant context.

![K-Sweep Chart](eval/k_sweep_chart.png)

## Performance Improvements

| Change | Why |
| --- | --- |
| Parallel doc fetching (`ThreadPoolExecutor`) | Eliminated ~18s of sequential wait |
| Batch embeddings (`batch_size=64`) | Faster than per-chunk encoding |
| Cached ChromaDB + tiktoken singletons | No re-init on every query |
| Real Groq streaming (`stream=True`) | First token in ~100–300ms |

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
