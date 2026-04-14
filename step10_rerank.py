"""
Cross-encoder re-ranking

Standard two-stage retrieval:
  Stage 1 — retrieve k=10 candidates using fast dense (bi-encoder) search
  Stage 2 — re-score each (query, document) pair with a cross-encoder,
             which reads the full pair together and produces a much more
             accurate relevance score
  Return the top 3 by cross-encoder score.

Why this improves on cosine similarity:
  A bi-encoder embeds query and document independently — it never sees them
  together. A cross-encoder reads both simultaneously, capturing interactions
  like "this query asks about Lambda timeouts connecting to RDS, and this
  chunk explains VPC configuration for Lambda" that bi-encoder similarity
  scores miss entirely.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Trained on MS MARCO passage retrieval. Fast (6-layer MiniLM),
  runs locally, no API cost.

Usage:
  python step10_rerank.py
  python step10_rerank.py --no-faithfulness
"""

import json
import time
import argparse
import chromadb
import groq
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
    CHROMA_DB_PATH, COLLECTION_NAME
)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
cross_encoder   = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
groq_client     = groq.Groq(api_key=GROQ_API_KEY)

CANDIDATE_K = 10   # retrieve this many with bi-encoder
FINAL_K     = 3    # keep this many after re-ranking


def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(name=COLLECTION_NAME)


def _groq_call_with_retry(messages, temperature, max_tokens, max_retries=5):
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


def retrieve_and_rerank(question: str, collection, final_k: int = FINAL_K):
    """
    1. Retrieve CANDIDATE_K chunks via bi-encoder cosine similarity.
    2. Score each (question, chunk) pair with the cross-encoder.
    3. Return top final_k by cross-encoder score.
    """
    query_embedding = embedding_model.encode(question, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=CANDIDATE_K,
        include=["documents", "metadatas"]
    )
    docs     = results["documents"][0]
    metas    = results["metadatas"][0]
    titles   = [m["title"] for m in metas]

    # Score pairs with cross-encoder
    pairs  = [(question, doc) for doc in docs]
    scores = cross_encoder.predict(pairs)

    # Sort by cross-encoder score descending, keep top final_k
    ranked = sorted(zip(scores, titles, docs), key=lambda x: x[0], reverse=True)
    top    = ranked[:final_k]

    top_titles  = [t for _, t, _ in top]
    top_context = "\n\n".join(d for _, _, d in top)
    return top_titles, top_context


def check_hit(retrieved_titles, expected_sources):
    for i, title in enumerate(retrieved_titles):
        if any(exp in title for exp in expected_sources):
            return True, i + 1
    return False, None


def generate_answer(question, context):
    prompt = (
        f"Answer the following question using ONLY the provided context. "
        f"Be concise.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer:"
    )
    response = _groq_call_with_retry(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content or ""


def score_faithfulness(question, context, answer):
    prompt = f"""You are an evaluation judge assessing whether an answer is grounded in the provided context.

QUESTION: {question}

CONTEXT (retrieved documentation):
{context[:2000]}

ANSWER:
{answer}

Score the answer from 1 to 5:
1 = Significant hallucinations
2 = Some fabricated details
3 = Mostly grounded with minor unsupported additions
4 = Well grounded, only minor inferences
5 = Fully grounded

Respond with ONLY a single integer 1-5."""
    response = _groq_call_with_retry(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )
    try:
        return max(1, min(5, int(response.choices[0].message.content.strip())))
    except (ValueError, AttributeError):
        return None


def run_evaluation(eval_path="eval_dataset.json", score_answers=True):
    with open(eval_path) as f:
        dataset = json.load(f)

    collection = get_collection()
    hits = 0
    reciprocal_ranks = []
    faith_scores = []
    results = []

    print(f"Re-rank evaluation — {len(dataset)} questions  "
          f"(candidates={CANDIDATE_K}, final_k={FINAL_K}, faithfulness={'on' if score_answers else 'off'})\n")
    print(f"{'#':<4} {'Question':<58} {'Ret':>5} {'Faith':>6}")
    print("-" * 78)

    for i, item in enumerate(dataset):
        question         = item["question"]
        expected_sources = item["expected_sources"]

        titles, context = retrieve_and_rerank(question, collection)
        hit, rank = check_hit(titles, expected_sources)

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
            "retrieved_titles": titles,
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
                faith_scores.append(faith_score)

        ret_str   = f"HIT@{rank}" if hit else "MISS"
        faith_str = f"{faith_score}/5" if faith_score is not None else "—"
        print(f"{i+1:<4} {question[:58]:<58} {ret_str:>5} {faith_str:>6}")
        results.append(result)

    n      = len(dataset)
    recall = hits / n
    mrr    = sum(reciprocal_ranks) / n
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else None

    print("\n" + "=" * 50)
    print("RE-RANKING SUMMARY")
    print("=" * 50)
    print(f"  Questions:        {n}")
    print(f"  Recall@{FINAL_K}:        {recall:.3f}  ({hits}/{n} hits)")
    print(f"  MRR@{FINAL_K}:           {mrr:.3f}")
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

    known_misses = {
        "hard-biz-003", "hard-biz-004", "hard-equiv-005",
        "hard-failure-001", "hard-failure-003"
    }
    print("\nPrevious misses:")
    for r in results:
        if r["id"] in known_misses:
            status = f"HIT@{r['rank']}" if r["hit"] else "MISS"
            print(f"  [{status}] {r['question'][:65]}")

    output = {
        "method": "cross-encoder re-ranking",
        "candidate_k": CANDIDATE_K,
        "final_k": FINAL_K,
        "summary": {
            "n": n,
            f"recall_at_{FINAL_K}": round(recall, 4),
            f"mrr_at_{FINAL_K}": round(mrr, 4),
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
    with open("eval_results_rerank.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nDetailed results saved to eval_results_rerank.json")
    return recall, mrr, avg_faith


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-faithfulness", action="store_true")
    parser.add_argument("--eval-path", default="eval_dataset.json")
    args = parser.parse_args()
    run_evaluation(eval_path=args.eval_path, score_answers=not args.no_faithfulness)
