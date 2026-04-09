"""
Document ingestion and chunking for the CloudDocs RAG system.

This module fetches AWS and Azure documentation pages, cleans HTML,
and splits content into meaningful chunks with metadata.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
from config import DATA_SOURCES, CHUNK_SIZE, CHUNK_OVERLAP
from urllib.parse import urljoin, urlparse

def clean_html_text(html_content):
    """
    Clean HTML content and extract readable text.

    CONCEPT: HTML parsing removes tags, scripts, and navigation elements,
    leaving only the main content text that's useful for RAG.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()

    # Get text and clean it up
    text = soup.get_text()

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)

    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks using sliding window technique.

    CONCEPT: Sliding window chunking maintains context between chunks
    by overlapping them. This prevents losing meaning at chunk boundaries.

    Example: Text "The quick brown fox jumps over the lazy dog"
    With chunk_size=20, overlap=5:
    Chunk 1: "The quick brown fox "
    Chunk 2: "fox jumps over the l"
    Chunk 3: "the lazy dog"
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_end = min(end + 100, len(text))
            sentence_end = text.rfind('.', end, search_end)
            if sentence_end != -1:
                end = sentence_end + 1
            else:
                # Look for word boundaries
                space_pos = text.rfind(' ', end, search_end)
                if space_pos != -1:
                    end = space_pos

        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap

        # Prevent infinite loop
        if start >= len(text):
            break

    return chunks

def fetch_document(url, title, category, provider):
    """
    Fetch a single document from the web.

    CONCEPT: Each document gets rich metadata (source, URL, category)
    that enables filtering and citation in the RAG system later.
    """
    try:
        print(f"Fetching: {title} ({url})")

        # Add headers to look like a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Clean the HTML content
        clean_text = clean_html_text(response.text)

        # Create chunks
        chunks = chunk_text(clean_text)

        # Create document metadata
        document = {
            "title": title,
            "url": url,
            "provider": provider,  # "aws" or "azure"
            "category": category,  # "compute", "storage", etc.
            "content": clean_text,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "total_chars": len(clean_text)
        }

        print(f"  ✓ Processed {len(chunks)} chunks ({len(clean_text)} chars)")
        return document

    except Exception as e:
        print(f"  ✗ Error fetching {url}: {str(e)}")
        return None

def main():
    """
    Main ingestion pipeline.

    CONCEPT: This creates a structured dataset of cloud documentation
    that will be embedded and stored for retrieval later.
    """
    print("🚀 Starting document ingestion...")
    print(f"Target chunk size: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP} chars\n")

    all_documents = []

    # Process AWS documentation
    print("📚 Processing AWS documentation...")
    for doc_config in DATA_SOURCES["aws"]:
        doc = fetch_document(
            doc_config["url"],
            doc_config["title"],
            doc_config["category"],
            "aws"
        )
        if doc:
            all_documents.append(doc)

        # Be respectful to the servers
        time.sleep(1)

    # Process Azure documentation
    print("\n☁️  Processing Azure documentation...")
    for doc_config in DATA_SOURCES["azure"]:
        doc = fetch_document(
            doc_config["url"],
            doc_config["title"],
            doc_config["category"],
            "azure"
        )
        if doc:
            all_documents.append(doc)

        # Be respectful to the servers
        time.sleep(1)

    # Save the processed documents
    output_file = "processed_documents.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)

    # Print summary
    total_chunks = sum(doc["chunk_count"] for doc in all_documents)
    total_chars = sum(doc["total_chars"] for doc in all_documents)

    print("\n✅ Ingestion complete!")
    print(f"📄 Documents processed: {len(all_documents)}")
    print(f"📦 Total chunks created: {total_chunks}")
    print(f"📊 Total characters: {total_chars:,}")
    print(f"💾 Saved to: {output_file}")

    # Show sample chunk
    if all_documents:
        sample_doc = all_documents[0]
        print(f"\n📖 Sample chunk from {sample_doc['title']}:")
        print("-" * 50)
        print(sample_doc["chunks"][0][:200] + "..." if len(sample_doc["chunks"][0]) > 200 else sample_doc["chunks"][0])

if __name__ == "__main__":
    main()