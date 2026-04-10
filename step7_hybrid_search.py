"""
Hybrid search evaluation: Dense-only vs BM25+Dense (RRF)

Background
----------
Swapping all-MiniLM-L6-v2 for BAAI/bge-base-en-v1.5 produced zero improvement
(both scored Recall=0.630, MRR=0.611). The bottleneck is not model size — it is
that dense embeddings encode *semantics*, so "S3", "GCP Storage", and "Azure Blob"
all land near each other in vector space because they are semantically equivalent
object-storage services. A better embedding model cannot fix this.

BM25 ignores semantics entirely and scores on exact token frequency. A query
containing "S3" gives "Amazon S3" a high BM25 score regardless of whether GCP
Storage is semantically similar. Combining BM25 with dense retrieval via
Reciprocal Rank Fusion (RRF) should recover the exact-keyword matches that dense
search misses.

Method
------
1. Load all chunks from ChromaDB (same index used by the existing pipeline).
2. Build a BM25 index over those chunks in memory.
3. For each eval question:
   - Dense: top-20 from ChromaDB cosine similarity
   - BM25:  top-20 by BM25 score
   - RRF:   merge both ranked lists -> take top-K
4. Measure Recall@K and MRR@K for dense-only vs hybrid.
5. Save results and generate a comparison chart.
"""

import json
import time
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from config import (
    EMBEDDING_MODEL, CHROMA_DB_PATH, COLLECTION_NAME
)

# ── Config ────────────────────────────────────────────────────────────────────

K_VALUES    = [3, 5]
EVAL_PATH   = "eval_dataset.json"
OUTPUT_JSON = "eval/hybrid_search_results.json"
OUTPUT_CHART= "eval/hybrid_search_chart.png"
CANDIDATE_N = 20   # candidates pulled from each retriever before RRF merge


# ── Load corpus from ChromaDB ─────────────────────────────────────────────────

def load_corpus():
    """Pull every chunk out of ChromaDB so we can build a BM25 index."""
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    total = collection.count()
    results = collection.get(
        limit=total,
        include=["documents", "metadatas", "embeddings"]
    )
    return results   # ids, documents, metadatas, embeddings


# ── BM25 index ────────────────────────────────────────────────────────────────

def build_bm25(documents):
    tokenized = [doc.lower().split() for doc in documents]
    return BM25Okapi(tokenized)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def dense_retrieve(query, collection, model, n):
    """Return top-n chunk indices by cosine similarity."""
    emb = model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[emb],
        n_results=n,
        include=["metadatas"]
    )
    return [m["title"] for m in results["metadatas"][0]]


def bm25_retrieve(query, bm25, all_titles, n):
    """Return top-n chunk titles by BM25 score."""
    scores = bm25.get_scores(query.lower().split())
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [all_titles[i] for i in ranked_indices[:n]]


def rrf_merge(dense_titles, bm25_titles, k_rrf=60):
    """Reciprocal Rank Fusion: combine two ranked lists into one."""
    scores = {}
    for rank, title in enumerate(dense_titles):
        scores[title] = scores.get(title, 0) + 1 / (rank + k_rrf)
    for rank, title in enumerate(bm25_titles):
        scores[title] = scores.get(title, 0) + 1 / (rank + k_rrf)
    return sorted(scores, key=scores.get, reverse=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

def score(retrieved_titles, expected_sources, k):
    """Return (hit, rank) for top-k of the retrieved list."""
    for i, title in enumerate(retrieved_titles[:k]):
        if any(src in title for src in expected_sources):
            return True, i + 1
    return False, None


def evaluate(dataset, collection, model, bm25, all_titles, k):
    dense_hits, dense_rr = 0, []
    hybrid_hits, hybrid_rr = 0, []

    for item in dataset:
        q        = item["question"]
        expected = item["expected_sources"]

        # Dense only
        d_titles = dense_retrieve(q, collection, model, CANDIDATE_N)
        hit, rank = score(d_titles, expected, k)
        dense_hits += hit
        dense_rr.append(1.0 / rank if hit else 0.0)

        # Hybrid (dense + BM25 via RRF)
        b_titles  = bm25_retrieve(q, bm25, all_titles, CANDIDATE_N)
        h_titles  = rrf_merge(d_titles, b_titles)
        hit, rank = score(h_titles, expected, k)
        hybrid_hits += hit
        hybrid_rr.append(1.0 / rank if hit else 0.0)

    n = len(dataset)
    return {
        "dense":  {"recall": round(dense_hits/n,  4), "mrr": round(sum(dense_rr)/n,  4), "hits": dense_hits,  "total": n},
        "hybrid": {"recall": round(hybrid_hits/n, 4), "mrr": round(sum(hybrid_rr)/n, 4), "hits": hybrid_hits, "total": n},
    }


# ── Chart ─────────────────────────────────────────────────────────────────────

def save_chart(results):
    import matplotlib.pyplot as plt
    import numpy as np

    k_vals   = K_VALUES
    methods  = ["dense", "hybrid"]
    labels   = ["Dense only", "Hybrid (BM25 + Dense)"]
    colors   = ["#4C72B0", "#2ca02c"]
    metrics  = ["recall", "mrr"]
    titles   = ["Recall@K", "MRR@K"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Retrieval: Dense-only vs Hybrid (BM25 + Dense RRF)",
                 fontsize=13, fontweight="bold")

    x     = np.arange(len(k_vals))
    width = 0.35

    for ax, metric, title in zip(axes, metrics, titles):
        for idx, (method, label) in enumerate(zip(methods, labels)):
            vals = [results[f"k{k}"][method][metric] for k in k_vals]
            bars = ax.bar(x + idx*width - width/2, vals, width,
                          label=label, color=colors[idx], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in k_vals])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs("eval", exist_ok=True)
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved -> {OUTPUT_CHART}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    print("Loading corpus from ChromaDB...")
    corpus  = load_corpus()
    docs    = corpus["documents"]
    titles  = [m["title"] for m in corpus["metadatas"]]
    print(f"  {len(docs)} chunks loaded")

    print("Building BM25 index...")
    bm25 = build_bm25(docs)

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)

    all_results = {}

    print(f"\n{'='*55}")
    print(f"  {'Method':<22} {'k':<5} {'Recall':>8} {'MRR':>8}")
    print(f"  {'-'*45}")

    for k in K_VALUES:
        res = evaluate(dataset, collection, model, bm25, titles, k)
        all_results[f"k{k}"] = res
        for method in ["dense", "hybrid"]:
            m = res[method]
            label = "Dense only" if method == "dense" else "Hybrid BM25+Dense"
            print(f"  {label:<22} {k:<5} {m['recall']:>8.3f} {m['mrr']:>8.3f}  ({m['hits']}/{m['total']} hits)")

    os.makedirs("eval", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved -> {OUTPUT_JSON}")

    save_chart(all_results)


if __name__ == "__main__":
    main()
