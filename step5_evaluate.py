"""
Evaluation pipeline for the CloudDocs RAG system.

Measures two things:
  1. Retrieval quality  — did the right source document appear in top-K results?
  2. Answer faithfulness — does the generated answer stay grounded in retrieved context?

Metrics:
  Recall@K   — fraction of questions where an expected source appeared in top-K
  MRR@K      — Mean Reciprocal Rank, rewards finding the right doc at rank 1 vs rank 5
  Faithfulness — LLM-as-judge score 1–5 averaged across all questions

CLI:
  python step5_evaluate.py                    # single run at k=8
  python step5_evaluate.py --k 5              # single run at k=5
  python step5_evaluate.py --no-faithfulness  # retrieval metrics only
  python step5_evaluate.py --sweep            # k-sweep across k=1,3,5,10
"""

import json
import argparse
import time
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import groq
from config import (
    GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
    CHROMA_DB_PATH, COLLECTION_NAME
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
groq_client = groq.Groq(api_key=GROQ_API_KEY)


def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(question, collection, k):
    """Embed question and return top-K titles + joined context string."""
    query_embedding = embedding_model.encode(question, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )
    titles = [m["title"] for m in results["metadatas"][0]]
    context = "\n\n".join(results["documents"][0])
    return titles, context


def check_hit(retrieved_titles, expected_sources):
    """Return (hit: bool, rank: int|None) for first matching expected source."""
    for i, title in enumerate(retrieved_titles):
        if any(expected in title for expected in expected_sources):
            return True, i + 1
    return False, None


def _groq_create(max_retries=5, **kwargs):
    """Call groq_client.chat.completions.create with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return groq_client.chat.completions.create(**kwargs)
        except groq.RateLimitError:
            wait = 2 ** attempt * 3  # 3, 6, 12, 24, 48s
            print(f"  Rate limit — waiting {wait}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)
    raise RuntimeError("Groq rate limit: max retries exceeded.")


def generate_answer(question, context):
    """Generate a grounded answer using retrieved context."""
    prompt = f"""Answer the following question using ONLY the provided context. Be concise.

CONTEXT:
{context}

QUESTION: {question}

Answer:"""
    response = _groq_create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400
    )
    return response.choices[0].message.content or ""


def score_faithfulness(question, context, answer):
    """
    LLM-as-judge: ask Groq to score the answer 1-5 for faithfulness.
    1 = heavily hallucinates, 5 = fully grounded in context.
    Returns int or None if parsing fails.
    """
    prompt = f"""You are an evaluation judge assessing whether an answer is grounded in the provided context.

QUESTION: {question}

CONTEXT (retrieved documentation):
{context[:2000]}

ANSWER:
{answer}

Score the answer from 1 to 5:
1 = Significant hallucinations — claims facts not in context
2 = Some fabricated details
3 = Mostly grounded with minor unsupported additions
4 = Well grounded, only minor inferences
5 = Fully grounded — every claim is supported by the context

Respond with ONLY a single integer 1-5 and nothing else."""

    response = _groq_create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5
    )
    try:
        return max(1, min(5, int(response.choices[0].message.content.strip())))
    except (ValueError, AttributeError):
        return None


def run_evaluation(eval_path="eval_dataset.json", k=8, score_answers=True, quiet=False):
    with open(eval_path, "r") as f:
        dataset = json.load(f)

    collection = get_collection()

    results = []
    hits = 0
    reciprocal_ranks = []
    faithfulness_scores = []

    if not quiet:
        print(f"Evaluating {len(dataset)} questions  (k={k}, faithfulness={'on' if score_answers else 'off'})\n")
        print(f"{'#':<4} {'Question':<60} {'Ret':>5} {'Faith':>6}")
        print("-" * 78)

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_sources = item["expected_sources"]

        retrieved_titles, context = retrieve(question, collection, k)
        hit, rank = check_hit(retrieved_titles, expected_sources)

        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        result = {
            "id": item["id"],
            "question": question,
            "category": item.get("category"),
            "expected_sources": expected_sources,
            "retrieved_titles": retrieved_titles,
            "hit": hit,
            "rank": rank,
        }

        faith_score = None
        if score_answers:
            answer = generate_answer(question, context)
            faith_score = score_faithfulness(question, context, answer)
            result["answer"] = answer
            result["faithfulness"] = faith_score
            if faith_score is not None:
                faithfulness_scores.append(faith_score)

        if not quiet:
            ret_str = f"HIT@{rank}" if hit else "MISS"
            faith_str = f"{faith_score}/5" if faith_score is not None else "—"
            print(f"{i+1:<4} {question[:60]:<60} {ret_str:>5} {faith_str:>6}")

        results.append(result)

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(dataset)
    recall = hits / n
    mrr = sum(reciprocal_ranks) / n
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None

    if not quiet:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"  Questions:       {n}")
        print(f"  Recall@{k}:       {recall:.3f}  ({hits}/{n} hits)")
        print(f"  MRR@{k}:          {mrr:.3f}")
        if avg_faith is not None:
            print(f"  Avg Faithfulness: {avg_faith:.2f} / 5.00")

        categories: dict[str, dict] = {}
        for item, result in zip(dataset, results):
            cat = item.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"hits": 0, "total": 0, "faith": []}
            categories[cat]["total"] += 1
            if result["hit"]:
                categories[cat]["hits"] += 1
            if result.get("faithfulness") is not None:
                categories[cat]["faith"].append(result["faithfulness"])

        print("\nRetrieval by category:")
        for cat, stats in sorted(categories.items()):
            pct = stats["hits"] / stats["total"]
            faith_avg = (
                f"  faith={sum(stats['faith'])/len(stats['faith']):.1f}"
                if stats["faith"] else ""
            )
            print(f"  {cat:<18} {stats['hits']}/{stats['total']}  ({pct:.0%}){faith_avg}")

        output = {
            "summary": {
                "n": n,
                f"recall_at_{k}": round(recall, 4),
                f"mrr_at_{k}": round(mrr, 4),
                "avg_faithfulness": round(avg_faith, 2) if avg_faith else None,
            },
            "per_category": {
                cat: {
                    "recall": round(s["hits"] / s["total"], 4),
                    "avg_faithfulness": round(sum(s["faith"]) / len(s["faith"]), 2) if s["faith"] else None,
                }
                for cat, s in categories.items()
            },
            "results": results,
        }
        with open("eval_results.json", "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nDetailed results saved to eval_results.json")

    return recall, mrr, avg_faith


# ── K-Sweep helpers ───────────────────────────────────────────────────────────

def _find_elbow(retrieval_rows):
    """Return the K row with the best recall-per-second efficiency."""
    return max(
        retrieval_rows,
        key=lambda x: x["recall"] / x["time_s"] if x["time_s"] > 0 else 0
    )


def _sweep_markdown(rows, best_time, best_result, best_balanced):
    lines = [
        "## K-Sweep Evaluation Results",
        "",
        "| K | Faithfulness | Recall@K | MRR@K | Avg Faith | Time (s) | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        notes = []
        if r["k"] == best_time["k"] and not r["faithfulness_scoring"]:
            notes.append("⚡ Best time")
        if r["k"] == best_result["k"] and r["faithfulness_scoring"]:
            notes.append("🏆 Best result")
        if r["k"] == best_balanced["k"] and not r["faithfulness_scoring"] and r["k"] != best_time["k"]:
            notes.append("⚖️ Best balanced")
        avg_f = f"{r['avg_faithfulness']:.2f}" if r["avg_faithfulness"] is not None else "—"
        lines.append(
            f"| {r['k']} | {'Yes' if r['faithfulness_scoring'] else 'No'} "
            f"| {r['recall']:.3f} | {r['mrr']:.3f} | {avg_f} | {r['time_s']} "
            f"| {', '.join(notes)} |"
        )
    composite = round(0.6 * best_result["recall"] + 0.4 * (best_result["avg_faithfulness"] or 0) / 5, 3)
    lines += [
        "",
        f"**⚡ Best time:** K={best_time['k']} (no faithfulness) — {best_time['time_s']}s, Recall={best_time['recall']:.3f}",
        f"**🏆 Best result:** K={best_result['k']} (with faithfulness) — composite {composite} (0.6×Recall + 0.4×Faith/5)",
        f"**⚖️ Best balanced:** K={best_balanced['k']} (no faithfulness) — highest recall-per-second",
        "",
        "![K-Sweep Chart](eval/k_sweep_chart.png)",
    ]
    return "\n".join(lines)


def _save_sweep_chart(rows, k_values):
    if not HAS_MATPLOTLIB:
        print("  matplotlib not installed — skipping chart (pip install matplotlib)")
        return
    os.makedirs("eval", exist_ok=True)

    ret_rows = sorted([r for r in rows if not r["faithfulness_scoring"]], key=lambda x: x["k"])
    full_rows = sorted([r for r in rows if r["faithfulness_scoring"]], key=lambda x: x["k"])

    ks = [r["k"] for r in ret_rows]
    recalls = [r["recall"] for r in ret_rows]
    mrrs = [r["mrr"] for r in ret_rows]
    faith_scores = [r["avg_faithfulness"] or 0 for r in full_rows]
    times_no_f = [r["time_s"] for r in ret_rows]
    times_f = [r["time_s"] for r in full_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("RAG K-Sweep Evaluation", fontsize=14, fontweight="bold")

    # Panel 1: Recall and MRR vs K
    axes[0].plot(ks, recalls, "o-", color="#2196F3", linewidth=2, markersize=8, label="Recall@K")
    axes[0].plot(ks, mrrs, "s--", color="#FF9800", linewidth=2, markersize=8, label="MRR@K")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Retrieval Quality vs K")
    axes[0].set_xticks(ks)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Faithfulness vs K
    axes[1].bar([str(k) for k in ks], faith_scores, color="#4CAF50", alpha=0.8)
    for i, v in enumerate(faith_scores):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Avg Faithfulness (1–5)")
    axes[1].set_title("Answer Faithfulness vs K")
    axes[1].set_ylim(0, 5)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Panel 3: Time vs K
    x = list(range(len(ks)))
    w = 0.35
    axes[2].bar([xi - w / 2 for xi in x], times_no_f, w, label="No faithfulness", color="#2196F3", alpha=0.8)
    axes[2].bar([xi + w / 2 for xi in x], times_f, w, label="With faithfulness", color="#F44336", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(k) for k in ks])
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Time (seconds)")
    axes[2].set_title("Evaluation Time vs K")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = "eval/k_sweep_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved to {path}")


def run_k_sweep(k_values=None, eval_path="eval_dataset.json"):
    """
    Run evaluation at k=1, 3, 5, 10 (with and without faithfulness).
    Times each run. Outputs:
      - eval_sweep_results.json  — raw numbers
      - eval_sweep_table.md      — markdown table ready to paste into README
      - eval/k_sweep_chart.png   — 3-panel chart (retrieval, faithfulness, time)
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    rows = []
    total = len(k_values) * 2

    for run_n, (k, score_answers) in enumerate(
        [(k, f) for k in k_values for f in [False, True]], start=1
    ):
        label = f"k={k}, faithfulness={'on' if score_answers else 'off'}"
        print(f"\n[{run_n}/{total}] {label} ...")
        t0 = time.time()
        recall, mrr, avg_faith = run_evaluation(
            eval_path=eval_path, k=k, score_answers=score_answers, quiet=True
        )
        elapsed = round(time.time() - t0, 1)
        rows.append({
            "k": k,
            "faithfulness_scoring": score_answers,
            "recall": round(recall, 4),
            "mrr": round(mrr, 4),
            "avg_faithfulness": avg_faith,
            "time_s": elapsed,
        })
        avg_f_str = f"{avg_faith:.2f}" if avg_faith is not None else "—"
        print(f"  Recall={recall:.3f}  MRR={mrr:.3f}  Faith={avg_f_str}  Time={elapsed}s")

    retrieval_rows = [r for r in rows if not r["faithfulness_scoring"]]
    full_rows = [r for r in rows if r["faithfulness_scoring"]]

    best_time = min(retrieval_rows, key=lambda x: x["time_s"])
    best_result = max(
        full_rows,
        key=lambda x: 0.6 * x["recall"] + 0.4 * (x["avg_faithfulness"] or 0) / 5
    )
    best_balanced = _find_elbow(retrieval_rows)

    md = _sweep_markdown(rows, best_time, best_result, best_balanced)
    _save_sweep_chart(rows, k_values)

    with open("eval_sweep_results.json", "w") as f:
        json.dump({
            "sweep_results": rows,
            "recommendations": {
                "best_time": best_time,
                "best_result": best_result,
                "best_balanced": best_balanced,
            }
        }, f, indent=2)

    with open("eval_sweep_table.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(md)
    print("\nLogs: eval_sweep_results.json  |  eval_sweep_table.md  |  eval/k_sweep_chart.png")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer quality")
    parser.add_argument("--k", type=int, default=8, help="Top-K chunks to retrieve (default: 8)")
    parser.add_argument(
        "--no-faithfulness", action="store_true",
        help="Skip answer generation and faithfulness scoring (faster, no extra API calls)"
    )
    parser.add_argument("--eval-path", default="eval_dataset.json", help="Path to eval dataset JSON")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run K-sweep across k=1,3,5,10 with and without faithfulness"
    )
    args = parser.parse_args()

    if args.sweep:
        run_k_sweep(eval_path=args.eval_path)
    else:
        run_evaluation(
            eval_path=args.eval_path,
            k=args.k,
            score_answers=not args.no_faithfulness
        )
