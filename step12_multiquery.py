"""
Multi-query retrieval via query reformulation + union pooling

The 3 remaining HyDE misses all share the same root cause: the original
query vocabulary doesn't surface the right document, and HyDE's single
hypothetical answer drifts toward the wrong service type.

Solution: generate N reformulations of the same question, each designed
to approach the concept from a different angle. Retrieve k candidates per
reformulation, take the union (deduplicated), track how many reformulations
retrieved each chunk ("retrieval confidence"), then re-rank the full union
against the original query using a cross-encoder.

Why this fixes Lambda/VPC:
  Original: "Lambda times out connecting to RDS — what is misconfigured?"
  Reformulation 1: "AWS Lambda access to private RDS database"
  Reformulation 2: "Lambda function network connectivity configuration"
  Reformulation 3: "Lambda VPC subnet security group RDS"
  → Reformulation 3 retrieves the VPC chunk that neither the original
    query nor HyDE's single pass surfaces.

Production reference: this pattern is used in Notion AI, Perplexity,
and Glean — often called "multi-query RAG" or "query fan-out retrieval."

Usage:
  python step12_multiquery.py
  python step12_multiquery.py --no-faithfulness
"""

import json
import argparse
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import EMBEDDING_MODEL
from rag_utils import (
    get_collection, groq_call, generate_answer, score_faithfulness, check_hit
)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
cross_encoder   = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

N_REFORMULATIONS = 3   # query variations to generate
K_PER_QUERY      = 6   # candidates per reformulation (union = up to 18)
FINAL_K          = 3   # keep after re-ranking

REFORMULATION_PROMPT = """\
Generate {n} different search queries that all ask for the same information as the original question below.
Each reformulation should use different vocabulary or frame the question from a different angle,
so that together they maximize coverage across a cloud documentation index.

For troubleshooting questions, include a reformulation that names the root-cause service or concept
(not just the symptoms). For plain-English questions, include a reformulation that uses the
technical service name.

Original question: {question}

Return ONLY a JSON array of {n} strings, no explanation:
["reformulation 1", "reformulation 2", "reformulation 3"]"""

# Failure taxonomy — applied post-hoc based on miss patterns
FAILURE_TAXONOMY = {
    "hard-failure-001": "multi_hop",        # Lambda → VPC: answer in different service
    "hard-biz-003":     "semantic_drift",   # HyDE drifts to wrong service type
    "hard-failure-003": "semantic_drift",   # same pattern
    "hard-biz-004":     "no_keyword",       # no VPC/network terms
    "hard-equiv-005":   "cross_vocabulary", # "same role as AWS IAM"
}


def generate_reformulations(question: str, n: int = N_REFORMULATIONS) -> list[str]:
    """Ask the LLM to rephrase the query N ways with different vocabulary."""
    prompt = REFORMULATION_PROMPT.format(n=n, question=question)
    response = groq_call(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,   # slight diversity
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    # Parse JSON array; fall back to original query if parsing fails
    try:
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:n]
    except (json.JSONDecodeError, ValueError):
        pass
    return [question] * n   # fallback


def retrieve_multiquery(question: str, collection, k_per: int = K_PER_QUERY,
                        final_k: int = FINAL_K):
    """
    1. Generate N reformulations of the question.
    2. Batch-encode all reformulations in a single forward pass.
    3. Retrieve k_per candidates per reformulation via bi-encoder.
    4. Union all candidates, deduplicate by document text, track
       how many reformulations retrieved each chunk (retrieval confidence).
    5. Re-rank the union against the ORIGINAL question using cross-encoder.
    6. Return top final_k.
    """
    queries = generate_reformulations(question)

    # Batch-encode all reformulations in a single forward pass
    query_embeddings = embedding_model.encode(
        queries, convert_to_numpy=True, batch_size=len(queries)
    )

    # Union candidates: doc_text -> {title, doc, hit_count}
    candidates: dict[str, dict] = {}
    for q_emb in query_embeddings:
        results = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k_per,
            include=["documents", "metadatas"]
        )
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if doc not in candidates:
                candidates[doc] = {"title": meta["title"], "doc": doc, "hit_count": 0}
            candidates[doc]["hit_count"] += 1

    # Re-rank the union against the ORIGINAL query
    pool      = list(candidates.values())
    pairs     = [(question, c["doc"]) for c in pool]
    ce_scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(ce_scores, pool), key=lambda x: x[0], reverse=True)
    top    = ranked[:final_k]

    top_titles  = [c["title"] for _, c in top]
    top_context = "\n\n".join(c["doc"] for _, c in top)
    top_conf    = [c["hit_count"] for _, c in top]  # retrieval confidence per doc
    return top_titles, top_context, queries, top_conf


def run_evaluation(eval_path="eval_dataset.json", score_answers=True):
    with open(eval_path) as f:
        dataset = json.load(f)

    collection = get_collection()
    hits = 0
    reciprocal_ranks: list[float] = []
    faith_scores: list[int] = []
    results: list[dict] = []

    print(f"Multi-query retrieval — {len(dataset)} questions  "
          f"(reformulations={N_REFORMULATIONS}, k_per={K_PER_QUERY}, "
          f"final_k={FINAL_K}, faithfulness={'on' if score_answers else 'off'})\n")
    print(f"{'#':<4} {'Question':<56} {'Ret':>5} {'Conf':>5} {'Faith':>6}")
    print("-" * 80)

    for i, item in enumerate(dataset):
        question         = item["question"]
        expected_sources = item["expected_sources"]

        titles, context, reformulations, confidences = retrieve_multiquery(
            question, collection
        )
        hit, rank = check_hit(titles, expected_sources)

        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        result = {
            "id":                   item["id"],
            "question":             question,
            "category":             item.get("category"),
            "expected_sources":     expected_sources,
            "retrieved_titles":     titles,
            "retrieval_confidence": confidences,
            "reformulations":       reformulations,
            "hit":                  hit,
            "rank":                 rank,
            "miss_reason":          FAILURE_TAXONOMY.get(item["id"]) if not hit else None,
        }

        faith_score = None
        if score_answers:
            answer      = generate_answer(question, context)
            faith_score = score_faithfulness(question, context, answer)
            result["answer"]       = answer
            result["faithfulness"] = faith_score
            if faith_score is not None:
                faith_scores.append(faith_score)

        ret_str   = f"HIT@{rank}" if hit else "MISS"
        conf_str  = f"{max(confidences)}/{N_REFORMULATIONS}" if confidences else "—"
        faith_str = f"{faith_score}/5" if faith_score is not None else "—"
        print(f"{i+1:<4} {question[:56]:<56} {ret_str:>5} {conf_str:>5} {faith_str:>6}")
        results.append(result)

    n         = len(dataset)
    recall    = hits / n
    mrr       = sum(reciprocal_ranks) / n
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else None

    print("\n" + "=" * 50)
    print("MULTI-QUERY SUMMARY")
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
        pct       = stats["hits"] / stats["total"]
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
            reason = f"  [{r['miss_reason']}]" if r.get("miss_reason") else ""
            print(f"  [{status}]{reason} {r['question'][:60]}")

    # Failure taxonomy summary
    miss_reasons = [r["miss_reason"] for r in results if r.get("miss_reason")]
    if miss_reasons:
        print("\nFailure taxonomy:")
        for reason in sorted(set(miss_reasons)):
            count = miss_reasons.count(reason)
            print(f"  {reason:<20} {count} miss(es)")

    output = {
        "method":              "multi-query reformulation + cross-encoder re-ranking",
        "n_reformulations":    N_REFORMULATIONS,
        "k_per_reformulation": K_PER_QUERY,
        "final_k":             FINAL_K,
        "summary": {
            "n":                    n,
            f"recall_at_{FINAL_K}": round(recall, 4),
            f"mrr_at_{FINAL_K}":    round(mrr, 4),
            "avg_faithfulness":     round(avg_faith, 2) if avg_faith else None,
        },
        "per_category": {
            cat: {
                "recall":           round(s["hits"] / s["total"], 4),
                "avg_faithfulness": round(sum(s["faith"]) / len(s["faith"]), 2)
                                    if s["faith"] else None,
            }
            for cat, s in categories.items()
        },
        "results": results,
    }
    with open("eval_results_multiquery.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nDetailed results saved to eval_results_multiquery.json")
    return recall, mrr, avg_faith


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-faithfulness", action="store_true")
    parser.add_argument("--eval-path", default="eval_dataset.json")
    args = parser.parse_args()
    run_evaluation(eval_path=args.eval_path, score_answers=not args.no_faithfulness)
