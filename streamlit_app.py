"""Streamlit Cloud entry point — handles secrets, DB init, and the chat UI."""

import os
import streamlit as st

# ── Load secrets into env before importing config ─────────────────────────────
# Streamlit Cloud stores secrets in st.secrets; copy them to env so config.py
# can find them via os.getenv() regardless of deployment environment.

for key in ("GROQ_API_KEY", "OPENAI_API_KEY"):
    if hasattr(st, "secrets") and key in st.secrets:
        os.environ[key] = st.secrets[key]

# ── Validate API key early — fail fast with a clear message ──────────────────

key = os.getenv("GROQ_API_KEY")
if not key:
    st.error(
        "❌ **GROQ_API_KEY not found!**\n\n"
        "**To fix:**\n"
        "1. Click **'Manage app'** (bottom right)\n"
        "2. Go to **Settings → Secrets**\n"
        "3. Add: `GROQ_API_KEY = \"gsk_your_key_here\"`\n"
        "4. Click **Save**\n\n"
        "Get your key: https://console.groq.com"
    )
    st.stop()

if not key.startswith("gsk_"):
    st.error(
        "❌ GROQ_API_KEY looks invalid — copy it exactly from the Groq Console "
        "without extra quotes or whitespace."
    )
    st.stop()

# ── Knowledge base init (runs once per deployment, cached by Streamlit) ───────

@st.cache_resource
def initialize_knowledge_base():
    """Ingest and embed docs on first load if the vector DB doesn't exist yet."""
    if os.path.exists("chroma_db") and os.path.exists("processed_documents.json"):
        return  # already initialised

    st.info("⏳ Initializing knowledge base on first load (2–3 minutes)...")
    try:
        st.info("📥 Fetching cloud documentation...")
        from step1_ingest import main as ingest_main
        ingest_main()
        st.success("✅ Ingestion complete")

        st.info("🔢 Creating embeddings...")
        from step2_embed_store import process_documents
        process_documents()
        st.success("✅ Knowledge base ready!")

    except Exception as e:
        st.error(f"❌ Error initializing knowledge base: {e}")
        st.error("Please try refreshing the page.")
        raise

initialize_knowledge_base()

# ── App setup ─────────────────────────────────────────────────────────────────

from step4_chat import RAGChat

st.set_page_config(page_title="CloudDocs RAG Assistant", page_icon="☁️", layout="wide")

if "chat" not in st.session_state:
    try:
        st.session_state.chat = RAGChat(quiet=True)
    except Exception as e:
        st.error(f"❌ Failed to initialize chat: {e}\nMake sure your GROQ_API_KEY is set in Secrets.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("☁️ CloudDocs RAG Knowledge Assistant")
st.markdown("Ask questions about AWS, Azure, and GCP cloud services")

with st.sidebar:
    st.header("🔍 Search Filters")
    provider_filter = st.selectbox("Filter by provider:", ["All", "aws", "azure", "gcp"])
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat.conversation_history = []
        st.rerun()

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new input
if prompt := st.chat_input("Ask about cloud services..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            filter_param = None if provider_filter == "All" else provider_filter
            response = st.session_state.chat.get_response(prompt, provider_filter=filter_param)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
