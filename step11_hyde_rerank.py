"""
HyDE + Cross-encoder re-ranking (combined pipeline)

  Stage 1 — HyDE: generate a hypothetical answer, embed it for retrieval
  Stage 2 — Re-rank: score (query, doc) pairs with a cross-encoder

HyDE improves recall by generating terminology-rich text for the embedding.
Re-ranking improves MRR by scoring the final candidates more accurately.
Together they address both failure modes from the 62-question eval set.

Usage:
  python step11_hyde_rerank.py
  python step11_hyde_rerank.py --no-faithfulness
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

CANDIDATE_K = 10
FINAL_K     = 3

_HYDE_PROMPT = (
    "Write a 2-3 sentence technical passage from cloud documentation "
    "that directly answers the following question. Use specific service "
    "names, technical terms, and cloud terminology.\n\n"
    "Question: {question}\n\nPassage:"
)


def generate_hypothetical_answer(question: str) -> str:
    response = groq_call(
        messages=[{"role": "user", "content": _HYDE_PROMPT.format(question=question)}],
        temperature=0.1,
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()


def retrieve_hyde_rerank(question: str,
                         collection) -> tuple[list[str], str, str]:
    # Stage 1: HyDE embedding → retrieve CANDIDATE_K
    hyp     = generate_hypothetical_answer(question)
    hyp_emb = embedding_model.encode(hyp, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[hyp_emb],
        n_results=CANDIDATE_K,
        include=["documents", "metadatas"],
    )
    docs   = results["documents"][0]
    titles = [m["title"] for m in results["metadatas"][0]]

    # Stage 2: cross-encoder re-ranking on the original query
    pairs  = [(question, doc) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, titles, docs), key=lambda x: x[0], reverse=True)
    top    = ranked[:FINAL_K]

    top_titles  = [t for _, t, _ in top]
    top_context = "\n\n".join(d for _, _, d in top)
    return top_titles, top_context, hyp


def run_evaluation(eval_path: str = "eval_dataset.json",
                   score_answers: bool = True):
    with open(eval_path) as f:
        dataset = json.load(f)

    collection = get_collection()
    hits = 0
    reciprocal_ranks: list[float] = []
    faith_scores: list[int] = []
    results: list[dict] = []

    print(f"HyDE + Re-rank eval — {len(dataset)} questions  "
          f"(candidates={CANDIDATE_K}, final_k={FINAL_K}, "
          f"faithfulness={'on' if score_answers else 'off'})\n")
    print(f"{'#':<4} {'Question':<58} {'Ret':>5} {'Faith':>6}")
    print("-" * 78)

    for i, item in enumerate(dataset):
        question         = item["question"]
        expected_sources = item["expected_sources"]

        titles, context, hyp = retrieve_hyde_rerank(question, collection)
        hit, rank = check_hit(titles, expected_sources)

        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        result = {
            "id":                  item["id"],
            "question":            question,
            "category":            item.get("category"),
            "expected_sources":    expected_sources,
            "retrieved_titles":    titles,
            "hypothetical_answer": hyp,
            "hit":                 hit,
            "rank":                rank,
        }

        faith_score = None
        if score_answers:
            answer      = generate_answer(question, context)
            faith_score = score_faithfulness(question, context, answer)
            result["answer"]      = answer
            result["faithfulness"] = faith_score
            if faith_score is not None:
                faith_scores.append(faith_score)

        ret_str   = f"HIT@{rank}" if hit else "MISS"
        faith_str = f"{faith_score}/5" if faith_score is not None else "—"
        print(f"{i+1:<4} {question[:58]:<58} {ret_str:>5} {faith_str:>6}")
        results.append(result)

    n         = len(dataset)
    recall    = hits / n
    mrr       = sum(reciprocal_ranks) / n
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else None

    print("\n" + "=" * 50)
    print("HYDE + RE-RANK SUMMARY")
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
        "hard-failure-001", "hard-failure-003",
    }
    print("\nPrevious misses:")
    for r in results:
        if r["id"] in known_misses:
            status = f"HIT@{r['rank']}" if r["hit"] else "MISS"
            print(f"  [{status}] {r['question'][:65]}")

    output = {
        "method":      "HyDE + cross-encoder re-ranking",
        "candidate_k": CANDIDATE_K,
        "final_k":     FINAL_K,
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
    with open("eval_results_hyde_rerank.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nDetailed results saved to eval_results_hyde_rerank.json")
    return recall, mrr, avg_faith


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-faithfulness", action="store_true")
    parser.add_argument("--eval-path",       default="eval_dataset.json")
    args = parser.parse_args()
    run_evaluation(eval_path=args.eval_path,
                   score_answers=not args.no_faithfulness)
