"""
Configuration for the CloudDocs RAG system.
Centralizes all settings, API keys, and hyperparameters.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def normalize_api_key(value):
    """Normalize API key string by trimming whitespace and surrounding quotes."""
    if not value:
        return None
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()
    return value

# Try to get API keys from environment or Streamlit secrets
def get_api_key(key_name):
    """Get API key from environment or Streamlit secrets."""
    # First try environment variable
    value = os.getenv(key_name)
    if value:
        return normalize_api_key(value)
    
    # Then try Streamlit secrets
    try:
        import streamlit as st
        if key_name in st.secrets:
            return normalize_api_key(st.secrets[key_name])
    except Exception:
        pass
    
    return None

# API Keys
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
GROQ_API_KEY = get_api_key("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "❌ GROQ_API_KEY not found!\n"
        "For Streamlit Cloud: Go to 'Manage app' → 'Settings' → 'Secrets' and add:\n"
        "GROQ_API_KEY = \"gsk_your_key_here\"\n"
        "For local development: Add to .env file:\n"
        "GROQ_API_KEY=gsk_your_key_here"
    )

if not GROQ_API_KEY.startswith("gsk_"):
    raise ValueError(
        "❌ GROQ_API_KEY looks invalid. "
        "Make sure you copied the key from https://console.groq.com exactly and did not include extra quotes or spaces."
    )

# Embedding Configuration (using free local model)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free SentenceTransformer model
EMBEDDING_DIMENSION = 384  # This model outputs 384-dimensional vectors

# LLM Configuration (using Groq for fast inference)
LLM_MODEL = "llama-3.1-8b-instant"  # Current Groq Llama model
TEMPERATURE = 0.1  # Low temperature for factual answers
MAX_TOKENS = 1000

# Vector Database Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "cloud_docs"

# Document Processing Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks for context continuity

# Retrieval Configuration
TOP_K_RETRIEVAL = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score for retrieval

# Data Sources (AWS + Azure documentation pages)
DATA_SOURCES = {
    "aws": [
        {
            "url": "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
            "title": "AWS Lambda",
            "category": "compute"
        },
        {
            "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "title": "Amazon S3",
            "category": "storage"
        },
        {
            "url": "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
            "title": "Amazon EC2",
            "category": "compute"
        },
        {
            "url": "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html",
            "title": "Amazon RDS",
            "category": "database"
        },
        {
            "url": "https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html",
            "title": "AWS Identity and Access Management",
            "category": "security"
        },
        {
            "url": "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html",
            "title": "Amazon VPC",
            "category": "networking"
        }
    ],
    "azure": [
        {
            "url": "https://docs.microsoft.com/en-us/azure/virtual-machines/",
            "title": "Azure Virtual Machines",
            "category": "compute"
        },
        {
            "url": "https://docs.microsoft.com/en-us/azure/storage/",
            "title": "Azure Storage",
            "category": "storage"
        },
        {
            "url": "https://docs.microsoft.com/en-us/azure/sql-database/",
            "title": "Azure SQL Database",
            "category": "database"
        },
        {
            "url": "https://docs.microsoft.com/en-us/azure/active-directory/",
            "title": "Azure Active Directory",
            "category": "security"
        },
        {
            "url": "https://docs.microsoft.com/en-us/azure/virtual-network/",
            "title": "Azure Virtual Network",
            "category": "networking"
        },
        {
            "url": "https://docs.microsoft.com/en-us/azure/functions/",
            "title": "Azure Functions",
            "category": "compute"
        }
    ],  
    "gcp": [
        {
            "url": "https://cloud.google.com/compute/docs",
            "title": "GCP Compute",
            "category": "compute"
        },
        {
            "url": "https://cloud.google.com/storage/docs",
            "title": "GCP Storage",
            "category": "storage"
        },
        {
            "url": "https://cloud.google.com/sql/docs",
            "title": "GCP Cloud SQL",
            "category": "database"
        },
        {
            "url": "https://cloud.google.com/iam/docs",
            "title": "GCP IAM",
            "category": "security"
        },
        {
            "url": "https://cloud.google.com/vpc/docs",
            "title": "GCP VPC",
            "category": "networking"
        },
        {
            "url": "https://cloud.google.com/functions/docs",
            "title": "GCP Cloud Functions",
            "category": "compute"
        }
    ]
}

# Chat Configuration
MAX_CONVERSATION_HISTORY = 10  # Number of message pairs to keep in memory
STREAMING_ENABLED = True  # Stream responses for better UX