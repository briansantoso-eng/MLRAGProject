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

## Evaluation Framework

To move beyond "it seems to work", a ground-truth evaluation set was built and three metrics were measured.

**What this shows:** `eval_dataset.json` contains 27 questions across 18 source documents (AWS, Azure, GCP) plus 3 cross-provider comparisons. Each question has a known expected source, enabling automated scoring without human review. Three metrics are tracked: **Recall@K** (did the right document appear in top-K?), **MRR@K** (how highly ranked was it?), and **LLM-as-judge Faithfulness** (does the generated answer stay grounded in retrieved context, scored 1–5 by a second Groq call?).

Baseline results at k=5 using `step5_evaluate.py`:

| Metric | Score |
| --- | --- |
| Recall@5 | 0.63 (17/27 hits) |
| MRR@5 | 0.61 |
| Avg Faithfulness (LLM-as-judge) | 3.7 / 5 |

**Retrieval by category:**

| Category | Recall@5 | Faithfulness |
| --- | --- | --- |
| Compute | 75% | 4.0 / 5 |
| Cross-provider | 100% | 4.3 / 5 |
| Storage | 50% | 2.8 / 5 |
| Database | 50% | 2.8 / 5 |
| Security | 50% | 4.0 / 5 |
| Networking | 50% | 4.3 / 5 |

**Key finding:** Cross-provider queries (e.g. "compare Lambda vs Azure Functions") hit 100% recall because either source satisfies the match. AWS-specific queries for S3, EC2, RDS, and VPC frequently retrieve semantically similar GCP or Azure equivalents instead — the embedding model treats equivalent services across clouds as near-identical vectors. Faithfulness is lower where retrieval misses, since the LLM is forced to answer from irrelevant context.

Run the eval yourself:

```bash
python step5_evaluate.py           # full eval with faithfulness scoring
python step5_evaluate.py --no-faithfulness  # retrieval metrics only
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
