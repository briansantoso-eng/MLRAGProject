# CloudDocs RAG System

A retrieval-augmented generation system built on real AWS, Azure, and GCP documentation. Ask questions, get grounded answers.

## Live Demo

**[https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/](https://mlragproject-e58tnq5unou9vsmswy4wrz.streamlit.app/)**

## Stack

- **LLM:** Groq Llama 3 8B
- **Embeddings:** SentenceTransformers all-MiniLM-L6-v2 (local, free)
- **Vector DB:** ChromaDB

## How it works

1. Scrapes real cloud documentation (AWS, Azure, GCP)
2. Chunks and embeds text into ChromaDB
3. Retrieves relevant passages on query
4. Generates grounded answers via Groq

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your Groq API key
python step1_ingest.py
python step2_embed_store.py
python step3_rag_query.py
python step4_chat.py
```

## Screenshot

![RAG Chat Screenshot](ss/ss1.png)
