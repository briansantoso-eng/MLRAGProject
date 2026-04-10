"""
RAG query pipeline for the CloudDocs RAG system.

This module retrieves relevant document chunks, constructs grounded prompts,
and generates answers using a modern LLM.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
import groq
import tiktoken
from config import (
    OPENAI_API_KEY, GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL, TEMPERATURE, MAX_TOKENS,
    CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_RETRIEVAL
)
import json

# Initialize clients
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
groq_client = groq.Groq(api_key=GROQ_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

try:
    _tiktoken_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
except Exception:
    _tiktoken_enc = None

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

def count_tokens(text, model=LLM_MODEL):
    """Count tokens in text using tiktoken."""
    if _tiktoken_enc:
        return len(_tiktoken_enc.encode(text))
    return len(text) // 4

def retrieve_relevant_chunks(query, collection, provider_filter=None, top_k=TOP_K_RETRIEVAL):
    """Retrieve most relevant chunks for a query."""
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()
    where_clause = {"provider": provider_filter} if provider_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []
    if results and results["documents"]:
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1 - distance
            chunk_info = {
                "text": doc,
                "metadata": metadata,
                "similarity": similarity,
                "source": f"{metadata['title']} ({metadata['provider'].upper()})"
            }
            retrieved_chunks.append(chunk_info)

    return retrieved_chunks, query_embedding

def build_rag_prompt(query, retrieved_chunks):
    """Build a prompt grounded in retrieved information."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"""[Source {i}] {chunk['source']}
Similarity: {chunk['similarity']:.3f}
Content: {chunk['text']}
""")

    context = "\n".join(context_parts)
    prompt = f"""You are a helpful cloud computing expert. Answer using ONLY the provided sources.

SOURCES:
{context}

QUESTION: {query}

Answer:"""
    return prompt

def generate_answer(prompt):
    """Generate answer using Groq LLM."""
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
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
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        else:
            raise Exception("Both Groq and OpenAI failed.")

def estimate_cost(prompt_tokens, completion_tokens):
    """Estimate API cost for the query."""
    input_cost = (prompt_tokens / 1_000_000) * 0.10
    output_cost = (completion_tokens / 1_000_000) * 0.10
    return input_cost + output_cost

def run_rag_query(query, provider_filter=None):
    """Complete RAG pipeline for a single query."""
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
        print(f"Similarity: {chunk['similarity']:.3f}")
        print(f"    📄 {chunk['source']}")
        print(f"    💬 {chunk['text'][:100]}...")

    rag_prompt = build_rag_prompt(query, retrieved_chunks)
    prompt_tokens = count_tokens(rag_prompt)

    print("\n🤖 Generating answer...")
    answer = generate_answer(rag_prompt)
    completion_tokens = count_tokens(answer)
    cost = estimate_cost(prompt_tokens, completion_tokens)

    print("\n💡 Answer:")
    print(answer)
    print("\n📊 Query Statistics:")
    print(f"   • Retrieved chunks: {len(retrieved_chunks)}")
    print(f"   • Prompt tokens: {prompt_tokens}")
    print(f"   • Completion tokens: {completion_tokens}")
    print(f"   • Estimated cost: ${cost:.6f}")

def main():
    """Demonstrate RAG with sample queries."""
    print("🚀 Testing RAG Query Pipeline")
    print("=" * 50)

    test_queries = [
        "How do I create a serverless function?",
        "What are the main differences between AWS Lambda and Azure Functions?",
        "How do I store and retrieve files in the cloud?",
    ]

    for query in test_queries:
        run_rag_query(query)

    print("\n" + "=" * 50)
    print("🔍 Testing Provider Filtering")
    run_rag_query("How do I create virtual machines?", "aws")
    run_rag_query("How do I create virtual machines?", "azure")

    print("\n✅ RAG testing complete!")

if __name__ == "__main__":
    main()