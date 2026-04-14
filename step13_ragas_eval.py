"""
RAGAS-style evaluation metrics

The standard RAG evaluation framework measures four things independently.
Three are implemented here using LLM-as-judge with prompts designed
for reliability (binary/numeric responses, not free text):

  Context Precision  — what fraction of retrieved chunks were actually
                       necessary to answer the question? Measures retrieval
                       noise. 1.0 = every retrieved chunk was useful.

  Context Recall     — did the retrieved context contain all the information
                       needed? Measures retrieval coverage. 1.0 = the context
                       contains everything in the reference answer.

  Answer Correctness — does the generated answer agree with the reference
                       answer on key facts? Measures end-to-end quality.
                       Scale 0.0–1.0.

Note on judge independence: this file intentionally uses the same LLM
(Groq Llama 3.1 8B) as the generation model. This is a known limitation —
a model cannot fully assess its own hallucinations. In production you would
use a different judge model (e.g., GPT-4o) or a trained reward model.
This limitation is documented explicitly in the results.

Usage:
  python step13_ragas_eval.py
  python step13_ragas_eval.py --sample 20   # run on 20 random questions
  python step13_ragas_eval.py --method hyde  # retrieval method (dense|hyde|multiquery)
"""

import json
import argparse
import random
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, LLM_MODEL
from rag_utils import get_collection, groq_call, generate_answer
from step12_multiquery import retrieve_multiquery

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# ── Prompts ───────────────────────────────────────────────────────────────────

CONTEXT_PRECISION_PROMPT = """\
You are evaluating retrieval quality. For each retrieved chunk below, determine
whether it contains information NECESSARY to answer the question.

Question: {question}

{chunks}

For each chunk, answer YES if it is necessary, NO if it is not.
Respond ONLY with a JSON array of YES/NO strings, one per chunk:
["YES", "NO", ...]"""

CONTEXT_RECALL_PROMPT = """\
You are evaluating whether retrieved context covers the reference answer completely.

Question: {question}
Reference Answer: {reference_answer}
Retrieved Context: {context}

Count the distinct factual claims in the reference answer.
For each claim, determine if it is supported by the retrieved context.
Respond ONLY with a JSON object: {{"supported": N, "total": M}}"""

ANSWER_CORRECTNESS_PROMPT = """\
Compare the generated answer to the reference answer for factual accuracy.

Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Score from 0.0 to 1.0:
  1.0 = all key facts agree
  0.5 = partial agreement, some facts correct, some missing or wrong
  0.0 = answers contradict or generated answer is completely wrong

Respond ONLY with a float between 0.0 and 1.0."""


def retrieve_dense(question: str, collection, k: int = 3):
    """Standard dense retrieval for comparison."""
    q_emb   = embedding_model.encode(question, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )
    docs    = results["documents"][0]
    titles  = [m["title"] for m in results["metadatas"][0]]
    context = "\n\n".join(docs)
    return titles, docs, context


def retrieve_hyde(question: str, collection, k: int = 3):
    """HyDE retrieval (imports logic inline to avoid circular deps)."""
    from step9_hyde import generate_hypothetical_answer
    hyp     = generate_hypothetical_answer(question)
    hyp_emb = embedding_model.encode(hyp, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[hyp_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )
    docs    = results["documents"][0]
    titles  = [m["title"] for m in results["metadatas"][0]]
    context = "\n\n".join(docs)
    return titles, docs, context


# ── RAGAS metrics ─────────────────────────────────────────────────────────────

def score_context_precision(question: str, docs: list[str]) -> float | None:
    """Fraction of retrieved chunks that were necessary to answer."""
    chunks_text = "\n\n".join(
        f"Chunk {i+1}: {doc[:400]}" for i, doc in enumerate(docs)
    )
    prompt   = CONTEXT_PRECISION_PROMPT.format(question=question, chunks=chunks_text)
    response = groq_call(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=40,
    )
    try:
        labels    = json.loads(response.choices[0].message.content.strip())
        yes_count = sum(1 for l in labels if str(l).upper() == "YES")
        return round(yes_count / len(labels), 4) if labels else None
    except (json.JSONDecodeError, ValueError, ZeroDivisionError):
        return None


def score_context_recall(question: str, reference_answer: str, context: str) -> float | None:
    """Fraction of reference answer claims supported by retrieved context."""
    prompt   = CONTEXT_RECALL_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        context=context[:3000]
    )
    response = groq_call(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=40,
    )
    try:
        obj = json.loads(response.choices[0].message.content.strip())
        return round(obj["supported"] / obj["total"], 4) if obj.get("total", 0) > 0 else None
    except (json.JSONDecodeError, KeyError, ZeroDivisionError, ValueError):
        return None


def score_answer_correctness(question: str, reference_answer: str,
                             generated_answer: str) -> float | None:
    """Agreement between generated answer and reference on key facts."""
    prompt   = ANSWER_CORRECTNESS_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer
    )
    response = groq_call(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    try:
        return round(max(0.0, min(1.0, float(response.choices[0].message.content.strip()))), 4)
    except (ValueError, AttributeError):
        return None


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_evaluation(eval_path="eval_dataset.json", method="multiquery",
                   sample: int | None = None):
    with open(eval_path) as f:
        dataset = json.load(f)

    if sample:
        random.seed(42)
        dataset = random.sample(dataset, min(sample, len(dataset)))

    collection = get_collection()
    results    = []
    scores     = {"context_precision": [], "context_recall": [], "answer_correctness": []}

    print(f"RAGAS eval — {len(dataset)} questions  (method={method})\n")
    print(f"Note: judge model = generation model ({LLM_MODEL}) — see docstring for limitation.\n")
    print(f"{'#':<4} {'Question':<50} {'CP':>5} {'CR':>5} {'AC':>5}")
    print("-" * 70)

    for i, item in enumerate(dataset):
        question         = item["question"]
        reference_answer = item.get("reference_answer", "")

        # Retrieve
        if method == "hyde":
            titles, docs, context = retrieve_hyde(question, collection)
        elif method == "multiquery":
            titles, context, _, _ = retrieve_multiquery(question, collection)
            # re-extract docs list from context for per-chunk precision
            docs = context.split("\n\n")
        else:
            titles, docs, context = retrieve_dense(question, collection)

        # Generate
        answer = generate_answer(question, context)

        # Score
        cp = score_context_precision(question, docs)
        cr = score_context_recall(question, reference_answer, context) if reference_answer else None
        ac = score_answer_correctness(question, reference_answer, answer) if reference_answer else None

        if cp is not None: scores["context_precision"].append(cp)
        if cr is not None: scores["context_recall"].append(cr)
        if ac is not None: scores["answer_correctness"].append(ac)

        result = {
            "id":                 item["id"],
            "question":           question,
            "category":           item.get("category"),
            "retrieved_titles":   titles,
            "generated_answer":   answer,
            "context_precision":  cp,
            "context_recall":     cr,
            "answer_correctness": ac,
        }
        results.append(result)

        cp_str = f"{cp:.2f}" if cp is not None else "—"
        cr_str = f"{cr:.2f}" if cr is not None else "—"
        ac_str = f"{ac:.2f}" if ac is not None else "—"
        print(f"{i+1:<4} {question[:50]:<50} {cp_str:>5} {cr_str:>5} {ac_str:>5}")

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    print("\n" + "=" * 50)
    print("RAGAS SUMMARY")
    print("=" * 50)
    print(f"  Method:             {method}")
    print(f"  Questions:          {len(dataset)}")
    print(f"  Context Precision:  {avg(scores['context_precision']):.3f}")
    print(f"  Context Recall:     {avg(scores['context_recall']):.3f}")
    print(f"  Answer Correctness: {avg(scores['answer_correctness']):.3f}")
    print(f"\n  Context Precision:  what fraction of retrieved chunks were necessary")
    print(f"  Context Recall:     what fraction of the reference answer was covered")
    print(f"  Answer Correctness: factual agreement with reference answer")
    print(f"\n  Limitation: judge = generator ({LLM_MODEL}) — may score itself generously")

    output = {
        "method":           method,
        "n":                len(dataset),
        "judge_model":      LLM_MODEL,
        "judge_limitation": "same model as generator — may not reliably detect own hallucinations",
        "summary": {
            "context_precision":  avg(scores["context_precision"]),
            "context_recall":     avg(scores["context_recall"]),
            "answer_correctness": avg(scores["answer_correctness"]),
        },
        "results": results,
    }
    out_path = f"eval_results_ragas_{method}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")
    return output["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dense", "hyde", "multiquery"],
                        default="multiquery")
    parser.add_argument("--sample", type=int, default=None,
                        help="Evaluate on a random subset of N questions")
    parser.add_argument("--eval-path", default="eval_dataset.json")
    args = parser.parse_args()
    run_evaluation(eval_path=args.eval_path, method=args.method, sample=args.sample)
