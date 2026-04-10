"""
Embedding model comparison: all-MiniLM-L6-v2 vs BAAI/bge-base-en-v1.5

Builds a separate ChromaDB collection for each model, runs retrieval
evaluation at k=3 and k=5, then generates a side-by-side comparison chart.
No faithfulness scoring (saves API calls — retrieval metrics tell the story).
"""

import json
import time
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "MiniLM (current)": "all-MiniLM-L6-v2",
    "BGE (new)":         "BAAI/bge-base-en-v1.5",
}
K_VALUES        = [3, 5]
DOCS_PATH       = "processed_documents.json"
EVAL_PATH       = "eval_dataset.json"
CHROMA_DB_PATH  = "./chroma_db_compare"
OUTPUT_JSON     = "eval/embedding_comparison.json"
OUTPUT_CHART    = "eval/embedding_comparison.png"


# ── Build collections ─────────────────────────────────────────────────────────

def build_collection(client, model_name, label, docs):
    """Embed all chunks and store in a named collection."""
    collection_name = label.replace(" ", "_").replace("(", "").replace(")", "").lower()

    # Drop and recreate for a clean run
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer(model_name)

    chunk_texts, metadatas, ids = [], [], []
    for doc in docs:
        for i, chunk in enumerate(doc.get("chunks", [])):
            chunk_texts.append(chunk)
            metadatas.append({
                "title":    doc["title"],
                "provider": doc["provider"],
                "url":      doc.get("url", ""),
                "category": doc.get("category", ""),
            })
            ids.append(f"{doc['title']}_{i}")

    print(f"  Encoding {len(chunk_texts)} chunks with {label}...")
    t0 = time.time()
    embeddings = model.encode(chunk_texts, batch_size=64,
                              convert_to_numpy=True, show_progress_bar=False)
    print(f"  Done in {time.time() - t0:.1f}s")

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=chunk_texts,
    )
    return collection, model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(collection, model, dataset, k):
    """Recall@k and MRR@k — no faithfulness."""
    hits, reciprocal_ranks = 0, []

    for item in dataset:
        query_emb = model.encode(item["question"],
                                 convert_to_numpy=True).tolist()
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["metadatas"]
        )
        titles = [m["title"] for m in results["metadatas"][0]]

        hit, rank = False, None
        for i, title in enumerate(titles):
            if any(src in title for src in item["expected_sources"]):
                hit, rank = True, i + 1
                break

        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(dataset)
    return {
        "recall":  round(hits / n, 4),
        "mrr":     round(sum(reciprocal_ranks) / n, 4),
        "hits":    hits,
        "total":   n,
    }


# ── Chart ─────────────────────────────────────────────────────────────────────

def save_chart(results):
    import matplotlib.pyplot as plt
    import numpy as np

    labels   = list(results.keys())          # model display names
    k_vals   = K_VALUES
    metrics  = ["recall", "mrr"]
    titles   = ["Recall@K", "MRR@K"]
    colors   = ["#4C72B0", "#DD8452"]        # blue = MiniLM, orange = BGE

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Embedding Model Comparison: MiniLM vs BGE", fontsize=14, fontweight="bold")

    x     = np.arange(len(k_vals))
    width = 0.35

    for ax, metric, title in zip(axes, metrics, titles):
        for idx, label in enumerate(labels):
            vals = [results[label][f"k{k}"][metric] for k in k_vals]
            bars = ax.bar(x + idx * width - width / 2, vals, width,
                          label=label, color=colors[idx], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
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
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved -> {OUTPUT_CHART}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs("eval", exist_ok=True)

    with open(DOCS_PATH, encoding="utf-8")  as f: docs    = json.load(f)
    with open(EVAL_PATH, encoding="utf-8")  as f: dataset = json.load(f)

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    all_results = {}

    for label, model_name in MODELS.items():
        print(f"\n{'='*55}")
        print(f"  Model: {label}  ({model_name})")
        print(f"{'='*55}")

        collection, model = build_collection(client, model_name, label, docs)

        label_results = {}
        for k in K_VALUES:
            metrics = evaluate(collection, model, dataset, k)
            label_results[f"k{k}"] = metrics
            print(f"  k={k}  Recall={metrics['recall']:.3f}  MRR={metrics['mrr']:.3f}  "
                  f"({metrics['hits']}/{metrics['total']} hits)")

        all_results[label] = label_results

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    print(f"{'Model':<22} {'k':<5} {'Recall':>8} {'MRR':>8}")
    print("-" * 45)
    for label, res in all_results.items():
        for k in K_VALUES:
            m = res[f"k{k}"]
            print(f"{label:<22} {k:<5} {m['recall']:>8.3f} {m['mrr']:>8.3f}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved -> {OUTPUT_JSON}")

    save_chart(all_results)


if __name__ == "__main__":
    main()
