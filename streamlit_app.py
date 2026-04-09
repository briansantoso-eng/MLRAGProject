import os
import sys
import streamlit as st

# Ensure vector DB is initialized on first run
@st.cache_resource
def initialize_knowledge_base():
    """Initialize the knowledge base if not already done."""
    if not os.path.exists("chroma_db"):
        st.info("⏳ Initializing knowledge base on first load (this may take 1-2 minutes)...")
        os.system("python step1_ingest.py")
        st.info("✅ Ingestion complete. Embedding documents...")
        os.system("python step2_embed_store.py")
        st.success("✅ Knowledge base ready!")

# Run initialization
initialize_knowledge_base()

# Now import and run the main app
from step4_chat import RAGChat

st.set_page_config(page_title="CloudDocs RAG Assistant", page_icon="☁️", layout="wide")

if 'chat' not in st.session_state:
    try:
        st.session_state.chat = RAGChat(quiet=True)
    except Exception as e:
        st.error(f"Failed to initialize chat: {e}")
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
