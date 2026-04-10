# CloudDocs RAG System

A retrieval-augmented generation system built on real AWS, Azure, and GCP documentation. Ask questions, get grounded answers.

## Live Demo

**[https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/](https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/)**

## Stack

- **LLM:** Groq Llama 3 8B
- **Embeddings:** SentenceTransformers all-MiniLM-L6-v2 (local, free)
- **Vector DB:** ChromaDB

## How it works

1. Scrapes real cloud documentation (AWS, Azure, GCP)
2. Chunks and embeds text into ChromaDB
3. Retrieves relevant passages on query
4. Generates grounded answers via Groq

## Engineering Decisions & Optimizations

This project went through several deliberate iterations to improve both runtime performance and retrieval quality.

### Performance optimizations (`step1`–`step4`)

| File | Change | Why |
| --- | --- | --- |
| `step1_ingest.py` | Sequential fetch + `sleep(1)` → `ThreadPoolExecutor(max_workers=5)` | 18 docs across 3 providers fetched concurrently; ~18s of dead wait eliminated |
| `step2_embed_store.py` | Per-chunk `model.encode()` loop → `model.encode(all_chunks, batch_size=64)` | SentenceTransformers is significantly faster in batch mode; one ChromaDB `add()` call per doc instead of N |
| `step3_rag_query.py` | New `ChromaDB` client created on every query → module-level singleton; `tiktoken.encoding_for_model()` called per token count → cached at import | Avoids re-initializing disk-backed DB and tokenizer on every question |
| `step4_chat.py` | Fake streaming (full response + word-by-word `sleep(0.05)`) → real Groq streaming API (`stream=True`) | First token appears in ~100–300ms instead of waiting for the full response |

### Evaluation methodology

Rather than just saying "it works", a proper ML evaluation was built in `step5_evaluate.py` with a hand-crafted 27-question eval set (`eval_dataset.json`) covering all three cloud providers and six categories.

Two metrics were measured:

- **Recall@K** — did the right source document appear in the top-K retrieved chunks?
- **Faithfulness** — does the generated answer stay grounded in the retrieved context? Scored 1–5 by a second Groq LLM-as-judge call.

### Finding the optimal K via a sweep

K (number of chunks passed to the LLM) is a direct tradeoff between retrieval coverage and cost/latency. To find the best value, a full K-sweep was run at K = 1, 3, 5, 10 — both with and without faithfulness scoring — timing every run.

| K | Faithfulness | Recall@K | MRR@K | Avg Faith | Time (s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | No | 0.593 | 0.593 | — | 1.9 | |
| 1 | Yes | 0.593 | 0.593 | 3.85 | 133.4 | |
| 3 | No | 0.630 | 0.611 | — | 0.5 | ⚡ Best time, ⚖️ Best balanced |
| 3 | Yes | 0.630 | 0.611 | 4.11 | 308.5 | 🏆 Best result |
| 5 | No | 0.630 | 0.611 | — | 0.9 | |
| 5 | Yes | 0.630 | 0.611 | 3.74 | 398.7 | |
| 10 | No | 0.630 | 0.611 | — | 0.6 | |
| 10 | Yes | 0.630 | 0.611 | 3.93 | 641.2 | |

**⚡ Best time:** K=3 (no faithfulness) — 0.5s, Recall=0.630
**🏆 Best result:** K=3 (with faithfulness) — composite 0.707 (0.6×Recall + 0.4×Faith/5)
**⚖️ Best balanced:** K=3 — highest recall-per-second efficiency

![K-Sweep Chart](eval/k_sweep_chart.png)

**Result: K=3 is the optimal value.** Recall plateaus completely from K=3 onwards — K=5 and K=10 retrieve the same documents with zero improvement. Faithfulness actually peaks at K=3 (4.11/5) and drops at K=5, because adding more chunks introduces irrelevant context that dilutes the LLM's answer. K=3 was set as the new system default.

**Root cause of misses:** AWS-specific queries (S3, EC2, RDS, IAM, VPC) frequently retrieve semantically equivalent GCP or Azure docs instead. The `all-MiniLM-L6-v2` embedding model treats equivalent cloud services across providers as near-identical vectors — "object storage" is "object storage" regardless of whether it's S3 or GCP Storage. Higher K does not fix this; it would require provider-aware metadata filtering at query time.

Run the sweep yourself:

```bash
python step5_evaluate.py --sweep             # full k-sweep (k=1,3,5,10)
python step5_evaluate.py --k 3               # single run at optimal K
python step5_evaluate.py --no-faithfulness   # retrieval metrics only, no API cost
```

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
