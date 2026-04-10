"""Embed document chunks with SentenceTransformers and store them in ChromaDB."""

import json
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, CHROMA_DB_PATH, COLLECTION_NAME

# Load model once at import time — avoids reloading on every call
model = SentenceTransformer(EMBEDDING_MODEL)


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def initialize_chroma():
    """Create (or open) the ChromaDB collection. Uses HNSW for fast ANN search."""
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"dimension": EMBEDDING_DIMENSION}
    )
    return client, collection


def load_existing_documents(collection):
    """Return the set of URLs already stored — used to skip re-embedding."""
    try:
        results = collection.get(include=["metadatas"])
        return {m["url"] for m in results["metadatas"] if "url" in m}
    except Exception:
        return set()


# ── Embedding helpers ─────────────────────────────────────────────────────────

def get_embedding(text):
    """Embed a single text string (used for similarity search tests)."""
    try:
        return model.encode(text, convert_to_numpy=True).tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_documents(documents_file="processed_documents.json"):
    """Batch-embed every chunk and upsert into ChromaDB."""
    print("🚀 Starting embedding and storage...")

    if not os.path.exists(documents_file):
        print(f"❌ {documents_file} not found — run step1_ingest.py first.")
        return

    with open(documents_file, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"📄 Loaded {len(documents)} documents")

    _, collection   = initialize_chroma()
    existing_urls   = load_existing_documents(collection)
    print(f"📚 {len(existing_urls)} documents already in database")

    total_chunks = new_chunks = 0

    for doc in documents:
        if doc["url"] in existing_urls:
            print(f"⏭️  Skipping {doc['title']} (already embedded)")
            total_chunks += doc["chunk_count"]
            continue

        print(f"🔄 Embedding {doc['title']} ({doc['chunk_count']} chunks)")

        chunk_texts = doc["chunks"]

        # Batch encode: one GPU/CPU pass for all chunks in this document
        embeddings = model.encode(chunk_texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)

        # Build parallel lists for a single batch insert into ChromaDB
        safe_title = doc["title"].replace(" ", "_")
        ids = [f"{doc['provider']}_{safe_title}_chunk_{i}" for i in range(len(chunk_texts))]
        metadatas = [
            {
                "url":          doc["url"],
                "title":        doc["title"],
                "provider":     doc["provider"],
                "category":     doc["category"],
                "chunk_index":  i,
                "total_chunks": doc["chunk_count"],
            }
            for i in range(len(chunk_texts))
        ]

        # One add() call per document instead of N individual calls
        collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=chunk_texts)

        new_chunks   += len(chunk_texts)
        total_chunks += len(chunk_texts)
        print(f"  ✅ Added {len(chunk_texts)} chunks")

    print("\n✅ Embedding complete!")
    print(f"📦 Total chunks in database: {total_chunks}")
    print(f"🆕 New chunks added:         {new_chunks}")
    print(f"💾 Database saved to:        {CHROMA_DB_PATH}")

    try:
        print(f"📊 Collection size:          {collection.count()} vectors")
    except Exception:
        pass


# ── Similarity search demo ────────────────────────────────────────────────────

def test_similarity_search(collection):
    """Show top-3 similar chunks for a few sample queries."""
    print("\n🔍 Testing similarity search...")
    queries = [
        "How do I create a virtual machine?",
        "What is serverless computing?",
        "How do I store files in the cloud?",
    ]
    for query in queries:
        print(f"\nQuery: '{query}'")
        if not (embedding := get_embedding(query)):
            continue
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        if results and results["documents"]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                print(f"  Similarity: {1 - dist:.3f}  |  {meta['title']} ({meta['provider']})")
                print(f"  {doc[:100]}...")


def main():
    process_documents()
    _, collection = initialize_chroma()
    test_similarity_search(collection)


if __name__ == "__main__":
    main()
