# RAG Concepts Deep Dive

This guide documents major concepts used by the CloudDocs RAG system. Each section includes practical explanations, code examples, and implementation insights for building retrieval-augmented systems.

## 1. Text Embeddings

### What are Embeddings?
Embeddings are numerical representations of text where similar meanings result in similar vectors, regardless of exact wording.

**Example:**
- "cat" → `[0.1, 0.8, -0.2, ...]` (1536 dimensions)
- "feline" → `[0.12, 0.79, -0.18, ...]` (very similar)
- "automobile" → `[-0.5, 0.1, 0.9, ...]` (very different)

### How They Work
Modern embedding models like `text-embedding-3-small` are trained on massive text datasets to learn semantic relationships. They use transformer architectures similar to GPT models.

### Code Example
```python
import openai

response = openai.embeddings.create(
    input="What is serverless computing?",
    model="text-embedding-3-small"
)
embedding = response.data[0].embedding  # 1536-dimensional vector
```

### Why 1536 Dimensions?
- Higher dimensions capture more nuanced relationships
- But require more storage and computation
- `text-embedding-3-small` balances quality and efficiency

## 2. Cosine Similarity

### The Math
Cosine similarity measures the angle between two vectors:

```
similarity = (A • B) / (|A| × |B|)
```

Where:
- `A • B` is the dot product
- `|A|` is the magnitude (length) of vector A

### Range and Meaning
- **1.0**: Vectors are identical (same direction)
- **0.0**: Vectors are perpendicular (unrelated)
- **-1.0**: Vectors are opposite (antonyms)

### Code Example
```python
import numpy as np

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
```

### Why Cosine vs Euclidean?
Cosine similarity is invariant to vector magnitude, focusing only on direction. This is perfect for embeddings where the "length" of the vector doesn't matter, only its "direction" (meaning).

## 3. Vector Databases & HNSW

### What is ChromaDB?
A specialized database for storing and searching high-dimensional vectors. Unlike traditional databases, it understands vector similarity.

### HNSW (Hierarchical Navigable Small World)
The algorithm ChromaDB uses for fast approximate nearest neighbor search:

1. **Hierarchical**: Multiple layers of graphs
2. **Navigable**: Each node connects to nearby nodes
3. **Small World**: Short paths between any two nodes

**Result:** Sub-second searches through millions of vectors

### Code Example
```python
import chromadb

# Create vector database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("docs")

# Store vectors with metadata
collection.add(
    ids=["chunk_1", "chunk_2"],
    embeddings=[[0.1, 0.2, ...], [0.15, 0.18, ...]],
    metadatas=[{"source": "aws"}, {"source": "azure"}],
    documents=["text chunk 1", "text chunk 2"]
)

# Search similar vectors
results = collection.query(
    query_embeddings=[[0.12, 0.19, ...]],
    n_results=5
)
```

## 4. Sliding Window Chunking

### The Problem
If you split text at arbitrary points, you lose context:

**Bad chunking:**
```
Chunk 1: "The quick brown fox"
Chunk 2: "jumps over the lazy dog"
```

**Good chunking:**
```
Chunk 1: "The quick brown fox "
Chunk 2: "fox jumps over the lazy"
Chunk 3: "lazy dog"
```

### Sliding Window Algorithm
1. Start at position 0
2. Take chunk of size N characters
3. Move start position by N - overlap
4. Repeat until end

### Code Example
```python
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Smart breaking at sentence boundaries
        if end < len(text):
            sentence_end = text.rfind('.', end - 100, end + 100)
            if sentence_end != -1:
                end = sentence_end + 1

        chunk = text[start:end].strip()
        chunks.append(chunk)

        start = end - overlap

    return chunks
```

### Why Overlap?
Prevents losing context at chunk boundaries. If important information spans a chunk boundary, the overlap ensures it's captured in both chunks.

## 5. Retrieval-Augmented Generation (RAG)

### The Architecture
```
User Query → Retrieval → Context + Query → LLM → Grounded Answer
```

### Why RAG Works
**Without RAG:** LLMs hallucinate when they don't know something
**With RAG:** LLMs get relevant context from your knowledge base

### Prompt Engineering for RAG
```python
def build_rag_prompt(query, retrieved_chunks):
    context = "\n".join([
        f"[Source {i+1}] {chunk['source']}\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])

    prompt = f"""Use ONLY the following sources to answer the question.

SOURCES:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Base your answer ONLY on the sources
- Cite sources when relevant
- Say "I don't have enough information" if needed

ANSWER:"""
    return prompt
```

### Retrieval Strategies
1. **Semantic Search**: Vector similarity (what we use)
2. **Keyword Search**: TF-IDF, BM25
3. **Hybrid Search**: Combine both
4. **Re-ranking**: Use cross-encoders for better accuracy

## 6. Token Counting & Cost Estimation

### What are Tokens?
LLMs don't process text character-by-character. They use tokens (sub-words):

- "serverless" → 1 token
- "computing" → 1 token
- "The quick brown fox" → ~5 tokens

### Why It Matters
- API costs are per token
- Context windows have token limits
- Accurate counting prevents overages

### Code Example
```python
import tiktoken

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Estimate cost
def estimate_cost(input_tokens, output_tokens):
    # GPT-4o-mini pricing
    input_cost = (input_tokens / 1_000_000) * 0.15
    output_cost = (output_tokens / 1_000_000) * 0.60
    return input_cost + output_cost
```

## 7. Conversation Memory

### The Challenge
Users ask follow-up questions with pronouns:
- "How do I create a VM?"
- "How does it compare to AWS?" (What is "it"?)

### Query Rewriting
Add conversation context to make queries self-contained:

```python
def rewrite_query(user_query, conversation_history):
    if seems_like_followup(user_query):
        context = get_recent_context(conversation_history)
        return f"Context: {context}\n\nQuestion: {user_query}"
    return user_query
```

### Memory Management
- Keep last N message pairs
- Summarize older messages if needed
- Clear memory on topic changes

## 8. Streaming Responses

### Why Stream?
- Better user experience (responses appear immediately)
- Prevents UI blocking
- Allows early termination if needed

### Code Example
```python
import openai

def stream_response(prompt):
    stream = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True  # Enable streaming
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)  # Print immediately

    print()  # New line at end
```

## 9. Metadata Filtering

### Why Filter?
Different users care about different providers:
- AWS-focused teams want AWS-only results
- Azure architects want Azure-only results

### Implementation
```python
# ChromaDB filtering
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"provider": "aws"}  # Filter to AWS only
)
```

### Advanced Filtering
- By category: `{"category": "compute"}`
- By recency: `{"date": {"$gt": "2024-01-01"}}`
- Complex queries: `{"$and": [{"provider": "aws"}, {"category": "storage"}]}`

## 10. Evaluation & Metrics

### Retrieval Metrics
- **Precision@K**: Fraction of top-K results that are relevant
- **Recall@K**: Fraction of all relevant docs in top-K
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks

### Generation Metrics
- **Groundedness**: How well answer matches retrieved context
- **Relevance**: How well answer matches the question
- **Factuality**: Absence of hallucinations

### Code Example
```python
def evaluate_retrieval(query, retrieved_docs, relevant_docs):
    retrieved_set = set(doc['id'] for doc in retrieved_docs)
    relevant_set = set(relevant_docs)

    # Precision@K
    precision = len(retrieved_set & relevant_set) / len(retrieved_docs)

    # Recall@K
    recall = len(retrieved_set & relevant_set) / len(relevant_set)

    return {"precision": precision, "recall": recall}
```

## 11. Cost Optimization

### Embedding Costs
- Use smaller models when possible (`text-embedding-3-small` vs `ada-002`)
- Cache embeddings for unchanged documents
- Incremental updates instead of full re-embedding

### Query Costs
- Retrieve fewer chunks (3-5 instead of 10-20)
- Use smaller LLMs for simple queries
- Implement query routing (simple questions → small model)

### Storage Costs
- Compress vectors if needed
- Use approximate search (HNSW) instead of exact
- Archive old/unused documents

## 12. Production Considerations

### Scalability
- **Million-scale**: ChromaDB with HNSW
- **Billion-scale**: Pinecone, Weaviate, Qdrant
- **Trillion-scale**: Custom distributed solutions

### Reliability
- **Error handling**: Retry failed API calls
- **Fallbacks**: Keyword search if vector search fails
- **Monitoring**: Track query latency, costs, accuracy

### Security
- **Data privacy**: Don't store sensitive documents
- **Access control**: Filter results by user permissions
- **Audit logging**: Track all queries and responses

## 13. Advanced RAG Patterns

### Multi-Query Retrieval
Generate multiple query variations for better recall:
```
Original: "How do I create a VM?"
Variations:
- "Create virtual machine AWS"
- "VM provisioning steps"
- "Launch EC2 instance"
```

### Re-ranking
Use cross-encoder models to re-order retrieved results for better accuracy.

### Agentic RAG
LLMs that can call tools, ask clarifying questions, or decompose complex queries.

### Hybrid Search
Combine vector similarity with traditional keyword search for best results.

---

This covers the core concepts. Each step in the project demonstrates these ideas in practice. Start with the basics, then explore advanced patterns as you build more complex RAG systems.