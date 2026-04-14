"""
Latency profiling — per-stage breakdown of the RAG pipeline

Measures wall-clock time for each stage of each retrieval method and
computes P50/P95 across N questions. Includes Groq server-side processing
time via the x-groq-processing-time response header and a hypothetical
cost estimate (based on Groq token pricing as of early 2025).

Stages tracked:
  embed_ms        — encode query (or hypothetical answer) with SentenceTransformer
  hyde_ms         — LLM call to generate hypothetical answer (HyDE only)
  reform_ms       — LLM call to generate query reformulations (multi-query only)
  vector_ms       — ChromaDB query (wall clock, one call per method variant)
  rerank_ms       — CrossEncoder.predict() call (re-rank methods only)
  generate_ms     — LLM call to generate final answer
  total_ms        — sum of all stages

Output:
  Console table with P50/P95 per stage per method
  latency_profile.json — raw per-question timings

Usage:
  python step14_latency_profile.py
  python step14_latency_profile.py --n 20          # profile on 20 questions
  python step14_latency_profile.py --method dense  # single method
"""

import json
import time
import argparse
import random
import statistics
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

# Groq token pricing (per 1M tokens, early-2025 public pricing for llama-3.1-8b-instant)
COST_PER_1M_INPUT_TOKENS  = 0.05   # USD
COST_PER_1M_OUTPUT_TOKENS = 0.08   # USD

GENERATION_PROMPT = """\
Answer the following question using ONLY the provided context. Be concise.

CONTEXT:
{context}

QUESTION: {question}

Answer:"""

HYDE_PROMPT = """\
Write a 2-3 sentence technical passage from cloud documentation that directly answers the following question.
Use specific service names, technical terms, and cloud terminology.

Question: {question}

Passage:"""

REFORM_PROMPT = """\
Generate 3 different search queries that all ask for the same information as the original question below.
Each reformulation should use different vocabulary or frame the question from a different angle,
so that together they maximize coverage across a cloud documentation index.

Original question: {question}

Return ONLY a JSON array of 3 strings, no explanation:
["reformulation 1", "reformulation 2", "reformulation 3"]"""


def _t():
    """Current time in milliseconds."""
    return time.perf_counter() * 1000


def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(name=COLLECTION_NAME)


def _groq_call_timed(messages, temperature, max_tokens, max_retries=5):
    """Call Groq, return (response, wall_ms, server_ms, input_tokens, output_tokens)."""
    delay = 6
    last_exc: Exception = RuntimeError("max_retries must be > 0")
    for attempt in range(max_retries):
        t0 = _t()
        try:
            resp = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            wall_ms = _t() - t0

            # Server-side processing time (excludes network latency)
            server_ms = None
            if hasattr(resp, "_raw_response") and resp._raw_response is not None:
                header = resp._raw_response.headers.get("x-groq-processing-time")
                if header:
                    try:
                        server_ms = float(header) * 1000
                    except ValueError:
                        pass

            in_tok  = getattr(resp.usage, "prompt_tokens",     0) or 0
            out_tok = getattr(resp.usage, "completion_tokens", 0) or 0
            return resp, wall_ms, server_ms, in_tok, out_tok

        except groq.RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    raise last_exc


def profile_dense(question: str, collection, k: int = 3) -> dict:
    timings: dict = {}
    total_in_tokens = 0
    total_out_tokens = 0

    t0 = _t()
    q_emb = embedding_model.encode(question, convert_to_numpy=True).tolist()
    timings["embed_ms"] = _t() - t0

    t0 = _t()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )
    timings["vector_ms"] = _t() - t0

    context = "\n\n".join(results["documents"][0])

    resp, wall_ms, server_ms, in_tok, out_tok = _groq_call_timed(
        messages=[{"role": "user", "content": GENERATION_PROMPT.format(
            context=context, question=question)}],
        temperature=0.1,
        max_tokens=400,
    )
    timings["generate_ms"]        = wall_ms
    timings["generate_server_ms"] = server_ms
    total_in_tokens  += in_tok
    total_out_tokens += out_tok

    timings["total_ms"]        = sum(v for k2, v in timings.items()
                                     if k2.endswith("_ms") and "server" not in k2)
    timings["input_tokens"]    = total_in_tokens
    timings["output_tokens"]   = total_out_tokens
    timings["est_cost_usd"]    = (
        total_in_tokens  / 1_000_000 * COST_PER_1M_INPUT_TOKENS +
        total_out_tokens / 1_000_000 * COST_PER_1M_OUTPUT_TOKENS
    )
    return timings


def profile_hyde(question: str, collection, k: int = 3) -> dict:
    timings: dict = {}
    total_in_tokens = 0
    total_out_tokens = 0

    resp, wall_ms, server_ms, in_tok, out_tok = _groq_call_timed(
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
        temperature=0.1,
        max_tokens=120,
    )
    timings["hyde_ms"]        = wall_ms
    timings["hyde_server_ms"] = server_ms
    hyp = resp.choices[0].message.content.strip()
    total_in_tokens  += in_tok
    total_out_tokens += out_tok

    t0 = _t()
    hyp_emb = embedding_model.encode(hyp, convert_to_numpy=True).tolist()
    timings["embed_ms"] = _t() - t0

    t0 = _t()
    results = collection.query(
        query_embeddings=[hyp_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )
    timings["vector_ms"] = _t() - t0

    context = "\n\n".join(results["documents"][0])

    resp, wall_ms, server_ms, in_tok, out_tok = _groq_call_timed(
        messages=[{"role": "user", "content": GENERATION_PROMPT.format(
            context=context, question=question)}],
        temperature=0.1,
        max_tokens=400,
    )
    timings["generate_ms"]        = wall_ms
    timings["generate_server_ms"] = server_ms
    total_in_tokens  += in_tok
    total_out_tokens += out_tok

    timings["total_ms"]        = sum(v for k2, v in timings.items()
                                     if k2.endswith("_ms") and "server" not in k2)
    timings["input_tokens"]    = total_in_tokens
    timings["output_tokens"]   = total_out_tokens
    timings["est_cost_usd"]    = (
        total_in_tokens  / 1_000_000 * COST_PER_1M_INPUT_TOKENS +
        total_out_tokens / 1_000_000 * COST_PER_1M_OUTPUT_TOKENS
    )
    return timings


def profile_multiquery(question: str, collection,
                       k_per: int = 6, final_k: int = 3) -> dict:
    timings: dict = {}
    total_in_tokens = 0
    total_out_tokens = 0

    resp, wall_ms, server_ms, in_tok, out_tok = _groq_call_timed(
        messages=[{"role": "user", "content": REFORM_PROMPT.format(question=question)}],
        temperature=0.3,
        max_tokens=200,
    )
    timings["reform_ms"]        = wall_ms
    timings["reform_server_ms"] = server_ms
    total_in_tokens  += in_tok
    total_out_tokens += out_tok

    raw = resp.choices[0].message.content.strip()
    try:
        queries = json.loads(raw)
        if not (isinstance(queries, list) and all(isinstance(q, str) for q in queries)):
            queries = [question] * 3
    except (json.JSONDecodeError, ValueError):
        queries = [question] * 3

    candidates: dict[str, dict] = {}
    total_vector_ms = 0.0
    total_embed_ms  = 0.0
    for q in queries:
        t0 = _t()
        q_emb = embedding_model.encode(q, convert_to_numpy=True).tolist()
        total_embed_ms += _t() - t0

        t0 = _t()
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=k_per,
            include=["documents", "metadatas"]
        )
        total_vector_ms += _t() - t0

        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if doc not in candidates:
                candidates[doc] = {"title": meta["title"], "doc": doc, "hit_count": 0}
            candidates[doc]["hit_count"] += 1

    timings["embed_ms"]  = total_embed_ms
    timings["vector_ms"] = total_vector_ms

    pool  = list(candidates.values())
    pairs = [(question, c["doc"]) for c in pool]
    t0 = _t()
    ce_scores = cross_encoder.predict(pairs)
    timings["rerank_ms"] = _t() - t0

    ranked  = sorted(zip(ce_scores, pool), key=lambda x: x[0], reverse=True)
    top     = ranked[:final_k]
    context = "\n\n".join(c["doc"] for _, c in top)

    resp, wall_ms, server_ms, in_tok, out_tok = _groq_call_timed(
        messages=[{"role": "user", "content": GENERATION_PROMPT.format(
            context=context, question=question)}],
        temperature=0.1,
        max_tokens=400,
    )
    timings["generate_ms"]        = wall_ms
    timings["generate_server_ms"] = server_ms
    total_in_tokens  += in_tok
    total_out_tokens += out_tok

    timings["total_ms"]        = sum(v for k2, v in timings.items()
                                     if k2.endswith("_ms") and "server" not in k2)
    timings["input_tokens"]    = total_in_tokens
    timings["output_tokens"]   = total_out_tokens
    timings["est_cost_usd"]    = (
        total_in_tokens  / 1_000_000 * COST_PER_1M_INPUT_TOKENS +
        total_out_tokens / 1_000_000 * COST_PER_1M_OUTPUT_TOKENS
    )
    return timings


def percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    return statistics.quantiles(data, n=100)[int(p) - 1]


def run_profile(eval_path="eval_dataset.json", n: int = 15,
                methods: list[str] | None = None):
    with open(eval_path) as f:
        dataset = json.load(f)

    random.seed(42)
    sample = random.sample(dataset, min(n, len(dataset)))
    collection = get_collection()

    if methods is None:
        methods = ["dense", "hyde", "multiquery"]

    stage_keys = {
        "dense":      ["embed_ms", "vector_ms", "generate_ms"],
        "hyde":       ["hyde_ms", "embed_ms", "vector_ms", "generate_ms"],
        "multiquery": ["reform_ms", "embed_ms", "vector_ms", "rerank_ms", "generate_ms"],
    }
    profile_fn = {
        "dense":      profile_dense,
        "hyde":       profile_hyde,
        "multiquery": profile_multiquery,
    }

    all_results = {}

    for method in methods:
        print(f"\nProfiling {method} — {len(sample)} questions …")
        timings_list = []

        for i, item in enumerate(sample):
            question = item["question"]
            t = profile_fn[method](question, collection)
            timings_list.append(t)
            total = t["total_ms"]
            print(f"  {i+1:>2}/{len(sample)}  total={total:6.0f}ms", end="\r")

        print(f"  Done.{' ' * 30}")

        # Compute P50/P95 per stage
        stats = {}
        for stage in stage_keys[method] + ["total_ms"]:
            vals = [t[stage] for t in timings_list if t.get(stage) is not None]
            if vals:
                stats[stage] = {
                    "p50": round(statistics.median(vals), 1),
                    "p95": round(percentile(vals, 95), 1),
                    "mean": round(sum(vals) / len(vals), 1),
                }

        cost_vals = [t["est_cost_usd"] for t in timings_list]
        stats["est_cost_usd_per_query"] = {
            "p50":  round(statistics.median(cost_vals) * 1000, 4),  # display in milli-USD
            "mean": round(sum(cost_vals) / len(cost_vals) * 1000, 4),
        }

        all_results[method] = {"stats": stats, "raw": timings_list}

    # Print comparison table
    print("\n" + "=" * 70)
    print("LATENCY PROFILE — P50 / P95 (milliseconds)")
    print("=" * 70)
    header = f"{'Stage':<20}"
    for m in methods:
        header += f"  {m:>20}"
    print(header)
    print("-" * 70)

    all_stages = []
    for m in methods:
        for s in stage_keys.get(m, []):
            if s not in all_stages:
                all_stages.append(s)
    all_stages.append("total_ms")

    for stage in all_stages:
        row = f"{stage:<20}"
        for m in methods:
            s = all_results[m]["stats"].get(stage)
            if s:
                row += f"  {s['p50']:>8.0f}ms/{s['p95']:.0f}"
            else:
                row += f"  {'—':>14}"
        print(row)

    print("\nFormat: P50 / P95")

    print("\n" + "=" * 70)
    print("COST ESTIMATE (milli-USD per query, Llama-3.1-8B on Groq)")
    print("=" * 70)
    for m in methods:
        c = all_results[m]["stats"].get("est_cost_usd_per_query", {})
        print(f"  {m:<12}  mean={c.get('mean', '?'):.4f}m$   "
              f"P50={c.get('p50', '?'):.4f}m$")
    print("  (1m$ = $0.001; 1000 queries ≈ $0.001–0.005 per method)")

    # Save raw results (strip raw timings to keep file small if > 20 questions)
    output = {method: {"stats": data["stats"]} for method, data in all_results.items()}
    with open("latency_profile.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSummary saved to latency_profile.json")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int,   default=15,
                        help="Number of questions to profile (default 15)")
    parser.add_argument("--method", type=str,   default=None,
                        help="Comma-separated methods: dense,hyde,multiquery")
    parser.add_argument("--eval-path", default="eval_dataset.json")
    args = parser.parse_args()

    methods = args.method.split(",") if args.method else None
    run_profile(eval_path=args.eval_path, n=args.n, methods=methods)
