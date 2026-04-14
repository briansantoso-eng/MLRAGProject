"""
Shared utilities for the RAG pipeline.

Centralises the functions that were copy-pasted across step5 / step9–step14:
  get_collection       — open the ChromaDB collection
  groq_call            — Groq chat completion with exponential-backoff retry
  generate_answer      — RAG answer generation
  score_faithfulness   — LLM-as-judge grounding score (1–5)
  check_hit            — retrieval hit / rank helper for eval loops
"""

import time
import chromadb
import groq
from chromadb.config import Settings

from config import GROQ_API_KEY, LLM_MODEL, CHROMA_DB_PATH, COLLECTION_NAME

groq_client = groq.Groq(api_key=GROQ_API_KEY)

_ANSWER_PROMPT = (
    "Answer the following question using ONLY the provided context. Be concise.\n\n"
    "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer:"
)

_FAITHFULNESS_PROMPT = (
    "You are an evaluation judge. Score whether the answer is grounded in the context.\n\n"
    "QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
    "Score 1–5 (5 = fully grounded, no unsupported claims). "
    "Respond with ONLY a single integer."
)


# ── Database ──────────────────────────────────────────────────────────────────

def get_collection():
    """Return the ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=COLLECTION_NAME)


# ── Groq API ──────────────────────────────────────────────────────────────────

def groq_call(messages: list[dict], temperature: float, max_tokens: int,
              max_retries: int = 5):
    """
    Call the Groq chat API with exponential-backoff retry on RateLimitError.

    Starts at a 6-second delay and doubles up to a 60-second ceiling.
    Raises the last RateLimitError if all retries are exhausted.
    """
    delay = 6
    last_exc: Exception = RuntimeError("max_retries must be > 0")
    for attempt in range(max_retries):
        try:
            return groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except groq.RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    raise last_exc


# ── Generation ────────────────────────────────────────────────────────────────

def generate_answer(question: str, context: str) -> str:
    """Generate a grounded answer from retrieved context."""
    response = groq_call(
        messages=[{"role": "user", "content": _ANSWER_PROMPT.format(
            context=context, question=question)}],
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content or ""


def score_faithfulness(question: str, context: str, answer: str) -> int | None:
    """
    Score whether an answer is grounded in the retrieved context.

    Returns an integer 1–5 (5 = fully grounded), or None on parse failure.
    """
    response = groq_call(
        messages=[{"role": "user", "content": _FAITHFULNESS_PROMPT.format(
            question=question,
            context=context[:2000],
            answer=answer,
        )}],
        temperature=0.0,
        max_tokens=5,
    )
    try:
        return max(1, min(5, int(response.choices[0].message.content.strip())))
    except (ValueError, AttributeError):
        return None


# ── Evaluation helpers ────────────────────────────────────────────────────────

def check_hit(retrieved_titles: list[str],
              expected_sources: list[str]) -> tuple[bool, int | None]:
    """
    Return (hit, rank) where rank is 1-based position of the first match.

    A title matches if any expected source string is a substring of it.
    Returns (False, None) when no match is found.
    """
    for rank, title in enumerate(retrieved_titles, start=1):
        if any(src in title for src in expected_sources):
            return True, rank
    return False, None
