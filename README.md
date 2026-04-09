# CloudDocs RAG System

CloudDocs RAG is a practical retrieval-augmented generation project built around real cloud provider documentation. It demonstrates how to combine document ingestion, vector search, and large language models to deliver grounded answers from AWS and Azure technical content.

## What this project does

- Ingests real AWS and Azure documentation
- Converts documentation into searchable vector embeddings
- Stores content in a local ChromaDB vector store
- Retrieves relevant passages for user queries
- Generates grounded answers using a modern LLM
- Supports conversational follow-up with memory and query rewriting

## Why it matters

Modern enterprise search and knowledge systems need answers that are both relevant and verifiable. This project shows how to:

- reduce hallucination by grounding responses in actual documentation
- enable cloud operations teams to query provider docs quickly
- speed up troubleshooting, onboarding, and architectural research
- build a reusable RAG pipeline for internal knowledge bases

## Real-world contribution

CloudDocs RAG is useful for teams that need:

- developer support for cloud architecture decisions
- fast access to multi-cloud documentation without manual search
- a proof-of-concept for integrating LLMs with real knowledge sources
- a low-cost pipeline for searchable technical content

## Key components

- `step1_ingest.py`: fetches and processes cloud docs into structured chunks
- `step2_embed_store.py`: creates and stores vector embeddings in ChromaDB
- `step3_rag_query.py`: runs retrieval-augmented queries with grounding
- `step4_chat.py`: provides a conversational interface with context handling
- `config.py`: central configuration for models, chunking, and retrieval

## Architecture

The system connects these layers:

- source documentation -> content ingestion
- text cleaning -> chunking with overlap
- vector embedding generation -> vector database storage
- similarity retrieval -> prompt construction
- LLM generation -> grounded response output

## Practical use cases

- internal knowledge search for platform engineering teams
- QA support for cloud operations and security teams
- training datasets for enterprise RAG systems
- evaluating cloud provider feature comparisons and best practices

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment:
   ```bash
   cp .env.example .env
   ```
3. Provide your OpenAI API key in `.env`

## How to run

Use the available scripts to build and query the system:

- `python step1_ingest.py`
- `python step2_embed_store.py`
- `python step3_rag_query.py`
- `python step4_chat.py`

## Benefits of this approach

- grounded answers from source documentation
- a clear vector search pipeline for cloud docs
- minimal cost with local embeddings and efficient LLM use
- extensible design for additional providers and content types

## Extending this project

This codebase is designed to scale beyond the initial cloud docs set:

- add GCP or other provider documentation
- support hybrid keyword + vector search
- build a web-based knowledge assistant
- add document freshness and source metadata filters
- enable multi-modal retrieval with diagrams or images

---

CloudDocs RAG is focused on turning documentation into reliable, reusable knowledge rather than teaching a single workflow step-by-step.
