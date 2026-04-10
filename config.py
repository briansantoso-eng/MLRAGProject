"""Configuration for the CloudDocs RAG system — all settings in one place."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API key helpers ───────────────────────────────────────────────────────────

def normalize_api_key(value):
    """Strip whitespace and surrounding quotes from a key string."""
    if not value:
        return None
    value = value.strip()
    if value[0] in ('"', "'") and value[0] == value[-1]:
        value = value[1:-1].strip()
    return value

def get_api_key(key_name):
    """Return a key from env var or Streamlit secrets, normalized."""
    if value := os.getenv(key_name):          # try env first (local dev)
        return normalize_api_key(value)
    try:
        import streamlit as st                 # fallback for Streamlit Cloud
        if key_name in st.secrets:
            return normalize_api_key(st.secrets[key_name])
    except Exception:
        pass
    return None

# ── Load and validate keys ────────────────────────────────────────────────────

OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
GROQ_API_KEY   = get_api_key("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "❌ GROQ_API_KEY not found!\n"
        "Streamlit Cloud: Manage app → Settings → Secrets → add GROQ_API_KEY = \"gsk_...\"\n"
        "Local: add GROQ_API_KEY=gsk_... to .env"
    )
if not GROQ_API_KEY.startswith("gsk_"):
    raise ValueError(
        "❌ GROQ_API_KEY looks invalid — copy it exactly from https://console.groq.com"
    )

# ── Model settings ────────────────────────────────────────────────────────────

EMBEDDING_MODEL     = "all-MiniLM-L6-v2"   # free, runs locally, 384-dim output
EMBEDDING_DIMENSION = 384

LLM_MODEL   = "llama-3.1-8b-instant"       # Groq Llama — fast and cheap
TEMPERATURE = 0.1                           # low = more factual, less creative
MAX_TOKENS  = 1000

# ── Vector database ───────────────────────────────────────────────────────────

CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "cloud_docs"

# ── Chunking ──────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 1000    # characters per chunk
CHUNK_OVERLAP = 200     # overlap preserves context across chunk boundaries

# ── Retrieval ─────────────────────────────────────────────────────────────────

TOP_K_RETRIEVAL      = 5    # chunks passed to LLM per query
SIMILARITY_THRESHOLD = 0.7  # minimum cosine similarity to include a result

# ── Data sources ──────────────────────────────────────────────────────────────

DATA_SOURCES = {
    "aws": [
        {"url": "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html", "title": "AWS Lambda",                         "category": "compute"},
        {"url": "https://docs.aws.amazon.com/s3/index.html",                 "title": "Amazon S3",                          "category": "storage"},
        {"url": "https://docs.aws.amazon.com/ec2/index.html",                "title": "Amazon EC2",                         "category": "compute"},
        {"url": "https://docs.aws.amazon.com/rds/index.html",                "title": "Amazon RDS",                         "category": "database"},
        {"url": "https://docs.aws.amazon.com/iam/index.html",                "title": "AWS Identity and Access Management", "category": "security"},
        {"url": "https://docs.aws.amazon.com/vpc/index.html",                "title": "Amazon VPC",                         "category": "networking"},
    ],
    "azure": [
        {"url": "https://docs.microsoft.com/en-us/azure/virtual-machines/", "title": "Azure Virtual Machines", "category": "compute"},
        {"url": "https://docs.microsoft.com/en-us/azure/storage/",          "title": "Azure Storage",          "category": "storage"},
        {"url": "https://docs.microsoft.com/en-us/azure/sql-database/",     "title": "Azure SQL Database",     "category": "database"},
        {"url": "https://docs.microsoft.com/en-us/azure/active-directory/", "title": "Azure Active Directory", "category": "security"},
        {"url": "https://docs.microsoft.com/en-us/azure/virtual-network/",  "title": "Azure Virtual Network",  "category": "networking"},
        {"url": "https://docs.microsoft.com/en-us/azure/functions/",        "title": "Azure Functions",        "category": "compute"},
    ],
    "gcp": [
        {"url": "https://cloud.google.com/compute/docs",   "title": "GCP Compute",         "category": "compute"},
        {"url": "https://cloud.google.com/storage/docs",   "title": "GCP Storage",         "category": "storage"},
        {"url": "https://cloud.google.com/sql/docs",       "title": "GCP Cloud SQL",       "category": "database"},
        {"url": "https://cloud.google.com/iam/docs",       "title": "GCP IAM",             "category": "security"},
        {"url": "https://cloud.google.com/vpc/docs",       "title": "GCP VPC",             "category": "networking"},
        {"url": "https://cloud.google.com/functions/docs", "title": "GCP Cloud Functions", "category": "compute"},
    ],
}

# ── Chat settings ─────────────────────────────────────────────────────────────

MAX_CONVERSATION_HISTORY = 10    # message pairs kept in memory
STREAMING_ENABLED        = True  # real-time token streaming via Groq API
