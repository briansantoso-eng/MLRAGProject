"""Single-turn RAG query pipeline — retrieve relevant chunks, build prompt, generate answer."""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
import groq
import tiktoken
from config import (
    OPENAI_API_KEY, GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL, TEMPERATURE, MAX_TOKENS,
    CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_RETRIEVAL,
)

# ── Module-level singletons (initialised once, reused across queries) ─────────

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
groq_client     = groq.Groq(api_key=GROQ_API_KEY)
openai_client   = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Cache tiktoken encoding — encoding_for_model() is slow if called per query
try:
    _tiktoken_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
except Exception:
    _tiktoken_enc = None

# Lazy-initialised ChromaDB collection — created on first query, reused after
_chroma_collection = None

def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        _chroma_collection = client.get_collection(name=COLLECTION_NAME)
    return _chroma_collection


# ── Core pipeline steps ───────────────────────────────────────────────────────

def count_tokens(text, model=LLM_MODEL):
    """Estimate token count using the cached tiktoken encoder."""
    return len(_tiktoken_enc.encode(text)) if _tiktoken_enc else len(text) // 4


def retrieve_relevant_chunks(query, collection, provider_filter=None, top_k=TOP_K_RETRIEVAL):
    """Embed query and return the top-K most similar chunks from ChromaDB."""
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"provider": provider_filter} if provider_filter else None,
        include=["documents", "metadatas", "distances"],
    )
    if not (results and results["documents"]):
        return [], query_embedding

    chunks = [
        {
            "text":       doc,
            "metadata":   meta,
            "similarity": 1 - dist,
            "source":     f"{meta['title']} ({meta['provider'].upper()})",
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    return chunks, query_embedding


def build_rag_prompt(query, retrieved_chunks):
    """Format retrieved chunks into a grounded prompt for the LLM."""
    context = "\n".join(
        f"[Source {i}] {c['source']}\nSimilarity: {c['similarity']:.3f}\nContent: {c['text']}\n"
        for i, c in enumerate(retrieved_chunks, 1)
    )
    return (
        f"You are a helpful cloud computing expert. Answer using ONLY the provided sources.\n\n"
        f"SOURCES:\n{context}\n"
        f"QUESTION: {query}\n\n"
        f"Answer:"
    )


def generate_answer(prompt):
    """Call Groq LLM, falling back to OpenAI if Groq fails."""
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with Groq API: {e}")
        if openai_client:
            print("Falling back to OpenAI...")
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        raise Exception("Both Groq and OpenAI failed.")


def estimate_cost(prompt_tokens, completion_tokens):
    """Estimate Groq API cost at $0.10 per million tokens."""
    return (prompt_tokens + completion_tokens) / 1_000_000 * 0.10


# ── Query runner ──────────────────────────────────────────────────────────────

def run_rag_query(query, provider_filter=None):
    """End-to-end RAG: retrieve → prompt → generate → print with stats."""
    print(f"\n🔍 Query: {query}")
    if provider_filter:
        print(f"📋 Filtered to: {provider_filter.upper()} only")

    collection = _get_collection()

    print("\n📚 Retrieving relevant information...")
    retrieved_chunks, _ = retrieve_relevant_chunks(query, collection, provider_filter)

    if not retrieved_chunks:
        print("❌ No relevant information found.")
        return

    print(f"✅ Found {len(retrieved_chunks)} relevant chunks:")
    for chunk in retrieved_chunks:
        print(f"  Similarity: {chunk['similarity']:.3f}  📄 {chunk['source']}")
        print(f"  💬 {chunk['text'][:100]}...")

    rag_prompt     = build_rag_prompt(query, retrieved_chunks)
    prompt_tokens  = count_tokens(rag_prompt)

    print("\n🤖 Generating answer...")
    answer            = generate_answer(rag_prompt)
    completion_tokens = count_tokens(answer)

    print(f"\n💡 Answer:\n{answer}")
    print(f"\n📊 Query Statistics:")
    print(f"   • Retrieved chunks:   {len(retrieved_chunks)}")
    print(f"   • Prompt tokens:      {prompt_tokens}")
    print(f"   • Completion tokens:  {completion_tokens}")
    print(f"   • Estimated cost:     ${estimate_cost(prompt_tokens, completion_tokens):.6f}")


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    print("🚀 Testing RAG Query Pipeline")
    print("=" * 50)

    for query in [
        "How do I create a serverless function?",
        "What are the main differences between AWS Lambda and Azure Functions?",
        "How do I store and retrieve files in the cloud?",
    ]:
        run_rag_query(query)

    print("\n" + "=" * 50)
    print("🔍 Testing Provider Filtering")
    run_rag_query("How do I create virtual machines?", "aws")
    run_rag_query("How do I create virtual machines?", "azure")

    print("\n✅ RAG testing complete!")


if __name__ == "__main__":
    main()
