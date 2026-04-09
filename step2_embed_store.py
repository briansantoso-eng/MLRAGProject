"""
Step 2: Embedding and Vector Storage

CONCEPTS YOU'LL LEARN:
- Text embeddings (converting text to numerical vectors)
- Cosine similarity and vector search
- ChromaDB vector database and HNSW indexing
- Incremental embedding (only process new documents)
- Metadata filtering for retrieval

This script converts text chunks into embeddings and stores them
in a vector database for fast similarity search.
"""

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import groq
from config import (
    GROQ_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    CHROMA_DB_PATH, COLLECTION_NAME
)
import numpy as np
import os

# Initialize SentenceTransformer model (free, runs locally)
model = SentenceTransformer(EMBEDDING_MODEL)

def get_embedding(text):
    """
    Convert text to embedding vector using SentenceTransformer.

    CONCEPT: SentenceTransformers provide free, high-quality embeddings
    that run locally without API calls or costs.
    """
    try:
        # Encode the text to get embeddings
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.

    CONCEPT: Cosine similarity measures the angle between vectors.
    Values range from -1 (opposite) to 1 (identical).
    For embeddings, higher values mean more similar content.

    Formula: similarity = (A • B) / (|A| × |B|)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

def initialize_chroma():
    """
    Initialize ChromaDB client and collection.

    CONCEPT: ChromaDB uses HNSW (Hierarchical Navigable Small World)
    indexing for fast approximate nearest neighbor search.
    This enables sub-second retrieval from millions of documents.
    """
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"dimension": EMBEDDING_DIMENSION}
    )

    return chroma_client, collection

def load_existing_documents(collection):
    """
    Get list of already processed document URLs.

    CONCEPT: Incremental processing avoids re-embedding unchanged documents,
    saving time and API costs when updating the knowledge base.
    """
    try:
        # Get all existing documents
        results = collection.get(include=["metadatas"])
        existing_urls = set()

        if results and results["metadatas"]:
            for metadata in results["metadatas"]:
                if "url" in metadata:
                    existing_urls.add(metadata["url"])

        return existing_urls
    except:
        return set()

def process_documents(documents_file="processed_documents.json"):
    """
    Main embedding and storage pipeline.
    """
    print("🚀 Starting embedding and storage...")

    # Load processed documents
    if not os.path.exists(documents_file):
        print(f"❌ Error: {documents_file} not found. Run step1_ingest.py first.")
        return

    with open(documents_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"📄 Loaded {len(documents)} documents")

    # Initialize ChromaDB
    chroma_client, collection = initialize_chroma()

    # Get existing documents to avoid reprocessing
    existing_urls = load_existing_documents(collection)
    print(f"📚 Found {len(existing_urls)} existing documents in database")

    # Process each document
    total_chunks = 0
    new_chunks = 0
    embedding_cost = 0

    for doc in documents:
        doc_url = doc["url"]

        # Skip if already processed
        if doc_url in existing_urls:
            print(f"⏭️  Skipping {doc['title']} (already processed)")
            total_chunks += doc["chunk_count"]
            continue

        print(f"🔄 Processing {doc['title']} ({doc['chunk_count']} chunks)")

        # Process each chunk
        for i, chunk in enumerate(doc["chunks"]):
            # Get embedding
            embedding = get_embedding(chunk)

            if embedding is None:
                print(f"  ❌ Failed to embed chunk {i+1}")
                continue

            # Calculate approximate cost (free for local embeddings)
            # Just tracking token count for reference
            token_estimate = len(chunk) * 0.02
            embedding_cost += 0  # Free!

            # Create unique ID for this chunk
            chunk_id = f"{doc['provider']}_{doc['title'].replace(' ', '_')}_chunk_{i}"

            # Prepare metadata
            metadata = {
                "url": doc["url"],
                "title": doc["title"],
                "provider": doc["provider"],
                "category": doc["category"],
                "chunk_index": i,
                "total_chunks": doc["chunk_count"]
            }

            # Store in ChromaDB
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[chunk]
            )

            new_chunks += 1
            total_chunks += 1

        print(f"  ✅ Added {doc['chunk_count']} chunks to database")

    # Print summary
    print("\n✅ Embedding complete!")
    print(f"📦 Total chunks in database: {total_chunks}")
    print(f"🆕 New chunks added: {new_chunks}")
    print(f"💰 Estimated embedding cost: ${embedding_cost:.4f}")
    print(f"💾 Database saved to: {CHROMA_DB_PATH}")

    # Show database stats
    try:
        count = collection.count()
        print(f"📊 Database contains {count} vectors")
    except:
        pass

def test_similarity_search(collection):
    """
    Demonstrate vector similarity search.

    CONCEPT: This shows how embeddings enable semantic search
    beyond simple keyword matching.
    """
    print("\n🔍 Testing similarity search...")

    test_queries = [
        "How do I create a virtual machine?",
        "What is serverless computing?",
        "How do I store files in the cloud?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Get query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            continue

        # Search for similar chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        if results and results["documents"]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                similarity = 1 - distance  # ChromaDB returns distance, convert to similarity
                print(f"Similarity: {similarity:.3f}")
                print(f"    From: {metadata['title']} ({metadata['provider']})")
                print(f"    Text: {doc[:100]}...")
                print()

def main():
    # Process documents
    process_documents()

    # Test similarity search
    chroma_client, collection = initialize_chroma()
    test_similarity_search(collection)

if __name__ == "__main__":
    main()