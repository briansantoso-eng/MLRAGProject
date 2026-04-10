"""
Evaluation: Baseline (no filter) vs Automatic Provider Detection

Why this branch exists
----------------------
Two prior experiments attempted to improve Recall@K beyond 0.630:
  1. BGE embedding model swap   -> no improvement (identical scores)
  2. BM25 hybrid search         -> made things worse (recall dropped at k=3)

Post-experiment analysis identified the root cause: cloud services across
providers (S3, GCP Storage, Azure Blob) are semantically identical and use
identical vocabulary. No retrieval algorithm can distinguish them without
knowing which provider the user is asking about.

This experiment tests the actual fix: detect the provider from query keywords
and scope ChromaDB retrieval with a metadata filter. The correct document is
already in the index — we just need to stop retrieving the wrong provider's
equivalent service alongside it.
"""

import json
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from provider_detector import detect_provider
from config import EMBEDDING_MODEL, CHROMA_DB_PATH, COLLECTION_NAME

K_VALUES    = [3, 5]
EVAL_PATH   = "eval_dataset.json"
OUTPUT_JSON = "eval/provider_detection_results.json"
OUTPUT_CHART= "eval/provider_detection_chart.png"


def retrieve(question, collection, model, k, provider_filter=None):
    emb = model.encode(question, convert_to_numpy=True).tolist()
    where = {"provider": provider_filter} if provider_filter else None
    results = collection.query(
        query_embeddings=[emb],
        n_results=k,
        where=where,
        include=["metadatas"]
    )
    return [m["title"] for m in results["metadatas"][0]]


def score(titles, expected_sources, k):
    for i, title in enumerate(titles[:k]):
        if any(src in title for src in expected_sources):
            return True, i + 1
    return False, None


def evaluate(dataset, collection, model, k, auto_detect=False):
    hits, rr = 0, []
    detections = {"correct": 0, "wrong": 0, "none": 0}

    for item in dataset:
        q        = item["question"]
        expected = item["expected_sources"]
        expected_provider = item.get("expected_provider")

        provider_filter = detect_provider(q) if auto_detect else None

        # Track detection accuracy
        if auto_detect:
            if provider_filter == expected_provider:
                detections["correct"] += 1
            elif provider_filter is None:
                detections["none"] += 1
            else:
                detections["wrong"] += 1

        titles = retrieve(q, collection, model, k, provider_filter)
        hit, rank = score(titles, expected, k)
        hits += hit
        rr.append(1.0 / rank if hit else 0.0)

    n = len(dataset)
    result = {
        "recall": round(hits / n, 4),
        "mrr":    round(sum(rr) / n, 4),
        "hits":   hits,
        "total":  n,
    }
    if auto_detect:
        result["detections"] = detections
    return result


def save_chart(results):
    import matplotlib.pyplot as plt
    import numpy as np

    k_vals  = K_VALUES
    methods = ["baseline", "auto_detect"]
    labels  = ["Baseline (no filter)", "Auto provider detection"]
    colors  = ["#4C72B0", "#2ca02c"]
    metrics = ["recall", "mrr"]
    titles  = ["Recall@K", "MRR@K"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Automatic Provider Detection vs Baseline",
                 fontsize=13, fontweight="bold")

    x, width = np.arange(len(k_vals)), 0.35

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


def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_results = {}

    print(f"\n{'='*58}")
    print(f"  {'Method':<28} {'k':<4} {'Recall':>8} {'MRR':>8}")
    print(f"  {'-'*50}")

    for k in K_VALUES:
        baseline    = evaluate(dataset, collection, model, k, auto_detect=False)
        auto_detect = evaluate(dataset, collection, model, k, auto_detect=True)
        all_results[f"k{k}"] = {"baseline": baseline, "auto_detect": auto_detect}

        for method, label, res in [
            ("baseline",    "Baseline (no filter)",    baseline),
            ("auto_detect", "Auto provider detection", auto_detect),
        ]:
            print(f"  {label:<28} {k:<4} {res['recall']:>8.3f} {res['mrr']:>8.3f}  ({res['hits']}/{res['total']} hits)")

        det = auto_detect["detections"]
        print(f"  {'Detection accuracy':<28}      correct={det['correct']}  none={det['none']}  wrong={det['wrong']}")
        print()

    os.makedirs("eval", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved -> {OUTPUT_JSON}")

    save_chart(all_results)


if __name__ == "__main__":
    main()
