"""Fetch cloud documentation pages, clean HTML, and split into overlapping chunks."""

import requests
from bs4 import BeautifulSoup
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DATA_SOURCES, CHUNK_SIZE, CHUNK_OVERLAP


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_html_text(html_content):
    """Strip HTML tags, scripts, and nav elements; return clean plain text."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Collapse whitespace: splitlines → strip each line → rejoin
    lines  = (line.strip() for line in soup.get_text().splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return " ".join(chunk for chunk in chunks if chunk)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Sliding-window chunking with sentence-boundary snapping.
    Overlap keeps context intact across chunk edges.
    """
    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Prefer breaking at a sentence end, then a word boundary
            search_end   = min(end + 100, len(text))
            sentence_end = text.rfind(".", end, search_end)
            if sentence_end != -1:
                end = sentence_end + 1
            elif (space := text.rfind(" ", end, search_end)) != -1:
                end = space

        if chunk := text[start:end].strip():
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ── Document fetching ─────────────────────────────────────────────────────────

def fetch_document(url, title, category, provider):
    """Fetch a URL, clean its HTML, and return a structured document dict."""
    try:
        print(f"Fetching: {title} ({url})")
        headers  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        clean_text = clean_html_text(response.text)
        chunks     = chunk_text(clean_text)

        print(f"  ✓ {len(chunks)} chunks ({len(clean_text)} chars)")
        return {
            "title":       title,
            "url":         url,
            "provider":    provider,
            "category":    category,
            "content":     clean_text,
            "chunks":      chunks,
            "chunk_count": len(chunks),
            "total_chars": len(clean_text),
        }

    except Exception as e:
        print(f"  ✗ Error fetching {url}: {e}")
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    """Fetch all configured docs in parallel and save chunks to JSON."""
    print("🚀 Starting document ingestion...")
    print(f"Chunk size: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP} chars\n")

    # Flatten provider → doc-list into a single list for parallel fetching
    all_configs = [
        (provider, cfg)
        for provider, docs in DATA_SOURCES.items()
        for cfg in docs
    ]

    all_documents = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_document, cfg["url"], cfg["title"], cfg["category"], provider): cfg["title"]
            for provider, cfg in all_configs
        }
        for future in as_completed(futures):
            if doc := future.result():
                all_documents.append(doc)

    # Persist to disk for step2 to consume
    output_file = "processed_documents.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)

    total_chunks = sum(d["chunk_count"] for d in all_documents)
    total_chars  = sum(d["total_chars"]  for d in all_documents)

    print("\n✅ Ingestion complete!")
    print(f"📄 Documents: {len(all_documents)}")
    print(f"📦 Chunks:    {total_chunks}")
    print(f"📊 Chars:     {total_chars:,}")
    print(f"💾 Saved to:  {output_file}")

    if all_documents:
        sample = all_documents[0]["chunks"][0]
        print(f"\n📖 Sample chunk from {all_documents[0]['title']}:")
        print("-" * 50)
        print(sample[:200] + "..." if len(sample) > 200 else sample)


if __name__ == "__main__":
    main()
