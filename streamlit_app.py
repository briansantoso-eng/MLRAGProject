import os
import sys
import streamlit as st

# Load Streamlit secrets into environment before importing other modules
if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
    os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Verify API key exists
if not os.getenv('GROQ_API_KEY'):
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

# Ensure vector DB is initialized on first run
@st.cache_resource
def initialize_knowledge_base():
    """Initialize the knowledge base if not already done."""
    if not os.path.exists("chroma_db") or not os.path.exists("processed_documents.json"):
        st.info("⏳ Initializing knowledge base on first load (this may take 2-3 minutes)...")
        
        try:
            # Step 1: Ingest documents
            st.info("📥 Fetching and processing cloud documentation...")
            from step1_ingest import main as ingest_main
            ingest_main()
            st.success("✅ Ingestion complete")
            
            # Step 2: Embed and store
            st.info("🔢 Creating embeddings and storing in vector database...")
            from step2_embed_store import process_documents
            process_documents()
            st.success("✅ Knowledge base ready!")
            
        except Exception as e:
            st.error(f"❌ Error initializing knowledge base: {e}")
            st.error("Please try refreshing the page")
            raise

# Run initialization
initialize_knowledge_base()

# Now import and run the main app
from step4_chat import RAGChat

st.set_page_config(page_title="CloudDocs RAG Assistant", page_icon="☁️", layout="wide")

if 'chat' not in st.session_state:
    try:
        st.session_state.chat = RAGChat(quiet=True)
        st.session_state.chat_initialized = True
    except Exception as e:
        st.error(f"❌ Failed to initialize chat: {str(e)}")
        st.error("Make sure your GROQ_API_KEY is set in Secrets")
        st.stop()

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("☁️ CloudDocs RAG Knowledge Assistant")
st.markdown("Ask questions about AWS, Azure, and GCP cloud services")

with st.sidebar:
    st.header("🔍 Search Filters")
    provider_filter = st.selectbox("Filter by provider:", ["All", "aws", "azure", "gcp"])
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat.conversation_history = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
