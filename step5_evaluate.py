"""
Evaluation pipeline for the CloudDocs RAG system.

Measures two things:
  1. Retrieval quality  — did the right source document appear in top-K results?
  2. Answer faithfulness — does the generated answer stay grounded in retrieved context?

Metrics:
  Recall@K   — fraction of questions where an expected source appeared in top-K
  MRR@K      — Mean Reciprocal Rank, rewards finding the right doc at rank 1 vs rank 5
  Faithfulness — LLM-as-judge score 1–5 averaged across all questions
"""

import json
import argparse
import time
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import groq
from config import (
    GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
    CHROMA_DB_PATH, COLLECTION_NAME
)

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


def _groq_call_with_retry(messages, temperature, max_tokens, max_retries=5):
    """Call Groq with exponential backoff on rate limit errors."""
    delay = 6
    last_exc: Exception = RuntimeError("max_retries must be > 0")
    for attempt in range(max_retries):
        try:
            return groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except groq.RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    raise last_exc


def generate_answer(question, context):
    """Generate a grounded answer using retrieved context."""
    prompt = f"""Answer the following question using ONLY the provided context. Be concise.

CONTEXT:
{context}

QUESTION: {question}

Answer:"""
    response = _groq_call_with_retry(
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

    response = _groq_call_with_retry(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5
    )
    try:
        return max(1, min(5, int(response.choices[0].message.content.strip())))
    except (ValueError, AttributeError):
        return None


def run_evaluation(eval_path="eval_dataset.json", k=5, score_answers=True):
    with open(eval_path, "r") as f:
        dataset = json.load(f)

    collection = get_collection()

    results = []
    hits = 0
    reciprocal_ranks = []
    faithfulness_scores = []

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

        ret_str = f"HIT@{rank}" if hit else "MISS"
        faith_str = f"{faith_score}/5" if faith_score is not None else "—"
        print(f"{i+1:<4} {question[:60]:<60} {ret_str:>5} {faith_str:>6}")

        results.append(result)

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(dataset)
    recall = hits / n
    mrr = sum(reciprocal_ranks) / n
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Questions:       {n}")
    print(f"  Recall@{k}:       {recall:.3f}  ({hits}/{n} hits)")
    print(f"  MRR@{k}:          {mrr:.3f}")
    if avg_faith is not None:
        print(f"  Avg Faithfulness: {avg_faith:.2f} / 5.00")

    # Per-category breakdown
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

    # Save full results
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

    print("\nDetailed results saved to eval_results.json")
    return recall, mrr, avg_faith


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer quality")
    parser.add_argument("--k", type=int, default=5, help="Top-K chunks to retrieve (default: 5)")
    parser.add_argument(
        "--no-faithfulness", action="store_true",
        help="Skip answer generation and faithfulness scoring (faster, no extra API calls)"
    )
    parser.add_argument("--eval-path", default="eval_dataset.json", help="Path to eval dataset JSON")
    args = parser.parse_args()

    run_evaluation(
        eval_path=args.eval_path,
        k=args.k,
        score_answers=not args.no_faithfulness
    )
