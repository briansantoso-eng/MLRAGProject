# CloudDocs RAG Project Memory

## Project Overview
This repository captures a complete RAG (Retrieval-Augmented Generation) system built on real AWS and Azure cloud documentation. It is designed to demonstrate how document ingestion, vector search, and LLM grounding can support practical cloud knowledge applications.

## Tech Stack
- **LLM**: Groq Llama 3 8B (fast inference, cost-effective)
- **Embeddings**: SentenceTransformers all-MiniLM-L6-v2 (free, local)
- **Vector DB**: ChromaDB with HNSW indexing
- **Web Scraping**: BeautifulSoup4 + requests
- **Config**: Python-dotenv for API keys

## Architecture
```
Real AWS/Azure Docs → Web Scraping → HTML Cleaning → Sliding Window Chunking → OpenAI Embeddings → ChromaDB → Cosine Similarity Retrieval → GPT-4o-mini → Grounded Answers
```

## Key Features
- **Cost-effective**: Total cost < $0.05 for complete system
- **Educational**: Every concept explained with code examples
- **Production-ready**: Error handling, streaming, conversation memory
- **Extensible**: Easy to add more cloud providers or document sources

## Purpose
- document a grounded RAG pipeline for cloud documentation
- show how to use vector search for provider docs
- provide a reproducible architecture for cloud knowledge retrieval
- illustrate real-world value by reducing hallucination and improving search relevance

## File Structure
- `step1_ingest.py` - Fetch and chunk cloud documentation
- `step2_embed_store.py` - Create embeddings and store in vector DB
- `step3_rag_query.py` - Test RAG queries with cost tracking
- `step4_chat.py` - Interactive chat with conversation memory
- `config.py` - All settings and API configurations
- `requirements.txt` - Python dependencies
- `README.md` - Complete setup and usage guide
- `CONCEPTS_GUIDE.md` - Deep dive into every RAG concept

## Current Status
✅ **Complete and tested**
- All 4 steps implemented and working
- Comprehensive documentation
- Cost optimization verified
- Error handling and edge cases covered

## Usage Instructions
```bash
pip install -r requirements.txt
cp .env.example .env  # Add OpenAI API key
python step1_ingest.py      # Fetch docs (~2 min)
python step2_embed_store.py # Embed chunks (~$0.001)
python step3_rag_query.py   # Test queries
python step4_chat.py        # Interactive chat
```

## Key Concepts Covered
- Text embeddings and cosine similarity
- Vector databases (ChromaDB, HNSW indexing)
- Sliding window chunking with overlap
- Retrieval-augmented generation
- Prompt engineering for grounded responses
- Token counting and cost estimation
- Conversation memory and query rewriting
- Streaming responses
- Metadata filtering

## Sample Queries
- "How do I create a serverless function?"
- "What are the differences between AWS Lambda and Azure Functions?"
- "How do I store files in the cloud?"
- "What are security best practices for cloud databases?"

## Cost Breakdown
- **Ingestion**: Free (web scraping)
- **Embeddings**: Free (local SentenceTransformers)
- **Queries**: ~$0.001 per question (Groq is much cheaper than OpenAI)
- **Total**: <$0.05 for complete project

## Future Extensions
- Add GCP documentation
- Implement hybrid search (keyword + vector)
- Add web UI with Streamlit
- Multi-modal support (diagrams, images)
- Agentic capabilities with tool calling

## Notes for Claude
- This project uses real cloud documentation, not synthetic data
- All concepts are documented in code comments and `CONCEPTS_GUIDE.md`
- The system is designed for practical clarity and cost efficiency
- Cost is kept extremely low to be accessible to everyone
- The implementation is structured to support incremental extension