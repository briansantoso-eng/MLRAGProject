# CloudDocs RAG System

A complete Retrieval-Augmented Generation (RAG) system built from scratch using real AWS and Azure documentation. This project teaches you every concept in modern RAG implementation while building something practical and cost-effective.

## 🎯 What You'll Learn

This 4-step project covers the complete RAG pipeline:

1. **Document Ingestion** - Web scraping, HTML cleaning, chunking algorithms
2. **Vector Embeddings** - Text-to-vector conversion, similarity search, ChromaDB
3. **RAG Querying** - Retrieval + generation, prompt engineering, cost optimization
4. **Interactive Chat** - Conversation memory, streaming responses, query rewriting

## 🏗️ Architecture

```
Web Pages → Ingestion → Chunking → Embeddings → Vector DB → Retrieval → Groq LLM → Answer
     ↓           ↓          ↓          ↓           ↓          ↓         ↓        ↓
   AWS/Azure   BeautifulSoup Sliding   OpenAI     ChromaDB   Cosine    Llama 3   Grounded
   Docs        Cleaning     Window     text-emb-3-small     Similarity           Response
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (free tier works)

### Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run the pipeline
python step1_ingest.py      # Fetch & chunk docs (~2 minutes)
python step2_embed_store.py # Create embeddings (~$0.001)
python step3_rag_query.py   # Test RAG queries
python step4_chat.py        # Interactive chat
```

## 📚 Step-by-Step Guide

### Step 1: Document Ingestion (`step1_ingest.py`)
**Concepts:** Web scraping, HTML parsing, sliding window chunking

This script fetches 12 real documentation pages from AWS and Azure, cleans the HTML, and splits them into overlapping chunks. Each chunk preserves metadata (source URL, provider, category) for later filtering.

**What it does:**
- Scrapes AWS Lambda, S3, EC2, RDS, IAM, VPC
- Scrapes Azure VMs, Storage, SQL DB, AAD, VNet, Functions
- Cleans HTML using BeautifulSoup
- Applies sliding window chunking (1000 chars, 200 overlap)
- Saves structured JSON with metadata

### Step 2: Embedding & Storage (`step2_embed_store.py`)
**Concepts:** Text embeddings, vector databases, cosine similarity, HNSW indexing

Converts text chunks into 1536-dimensional vectors using OpenAI's `text-embedding-3-small` model, then stores them in ChromaDB for fast similarity search.

**What it does:**
- Embeds each chunk using OpenAI API
- Stores vectors in ChromaDB with metadata
- Implements incremental processing (skips existing docs)
- Tests similarity search with sample queries

### Step 3: RAG Query Pipeline (`step3_rag_query.py`)
**Concepts:** Retrieval-augmented generation, prompt engineering, token counting

Demonstrates the complete RAG loop: retrieve relevant chunks, feed them to GPT-4o-mini, generate grounded answers with cost tracking.

**What it does:**
- Takes natural language queries
- Retrieves top-5 similar chunks
- Builds grounded prompts with sources
- Generates answers using GPT-4o-mini
- Shows similarity scores and cost estimates
- Supports provider filtering (AWS-only, Azure-only)

### Step 4: Interactive Chat (`step4_chat.py`)
**Concepts:** Conversation memory, query rewriting, streaming responses

Creates an interactive chat interface that maintains conversation history and handles follow-up questions intelligently.

**What it does:**
- Maintains conversation context
- Rewrites vague follow-ups ("how does it work?") with history
- Streams responses for better UX
- Supports provider filtering
- Memory management (auto-trims old messages)

## 💰 Cost Breakdown

Using free local embeddings + Groq for inference:

- **Ingestion:** Free (just bandwidth)
- **Embeddings:** Free (runs locally with SentenceTransformers)
- **Queries:** ~$0.001 per question (Groq inference only)
- **Chat:** ~$0.005 per conversation

**Total cost for full project:** <$0.05

## 🎮 Sample Queries to Try

```
"How do I create a serverless function?"
"What are the differences between AWS Lambda and Azure Functions?"
"How do I store files in the cloud?"
"What are security best practices for cloud databases?"
"How do I set up virtual networks?"
```

## 🔧 Configuration

All settings are in `config.py`:

- **Embedding:** `all-MiniLM-L6-v2` (free local SentenceTransformer)
- **LLM:** `llama-3.1-8b-instant` (Groq's fast Llama 3.1 model)
- **Chunking:** 1000 chars with 200 char overlap
- **Retrieval:** Top 5 chunks, cosine similarity
- **Chat:** 10 message memory, streaming enabled

## 📖 Key Concepts Explained

### Text Embeddings
Text is converted to numerical vectors where similar meanings have similar vectors. "Cat" and "feline" will be close in vector space even though they share no common letters.

### Cosine Similarity
Measures the angle between vectors. Values from -1 (opposite) to 1 (identical). For RAG, higher similarity means more relevant content.

### Sliding Window Chunking
Splits text into overlapping chunks to maintain context. Prevents losing meaning at arbitrary cut points.

### Retrieval-Augmented Generation
Combines information retrieval (finding relevant docs) with generation (synthesizing answers). Prevents hallucination by grounding responses in actual documentation.

### Vector Databases
Specialized databases for storing and searching high-dimensional vectors. Use HNSW indexing for sub-second searches through millions of documents.

## 🚨 Troubleshooting

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**OpenAI API errors:**
- Check your API key in `.env`
- Ensure you have credits in your OpenAI account

**No documents found:**
- Run `step1_ingest.py` first
- Check `processed_documents.json` was created

**Empty responses:**
- Run `step2_embed_store.py` to create the vector database
- Check `chroma_db` folder exists

## 🔄 Extending the Project

Ideas for enhancement:
- Add more cloud providers (GCP, DigitalOcean)
- Implement hybrid search (keyword + vector)
- Add document freshness checking
- Create a web UI with Streamlit
- Add multi-modal support (images, diagrams)
- Implement agentic RAG with tool calling

## 📚 Further Reading

- [Retrieval-Augmented Generation paper](https://arxiv.org/abs/2005.11401)
- [OpenAI Embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB documentation](https://docs.trychroma.com/)
- [BeautifulSoup documentation](https://www.crummy.com/software/BeautifulSoup/)

---

**Built for learning, optimized for cost, powered by real documentation.**