"""Local Streamlit web UI — wraps RAGChat in a chat interface."""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step4_chat import RAGChat

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="CloudDocs RAG Assistant", page_icon="☁️", layout="wide")

# ── Session state init ────────────────────────────────────────────────────────

if "chat" not in st.session_state:
    try:
        st.session_state.chat = RAGChat(quiet=True)
        st.session_state.chat_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize RAG Chat: {e}")
        st.session_state.chat_initialized = False
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.title("☁️ CloudDocs RAG Knowledge Assistant")
    st.markdown("Ask questions about AWS, Azure, and GCP cloud services")

    with st.sidebar:
        st.header("🔍 Search Filters")
        provider_filter = st.selectbox(
            "Filter by provider:",
            ["All", "aws", "azure", "gcp"],
            help="Limit search to a specific cloud provider",
        )
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat.conversation_history = []
            st.rerun()

    # Render existing chat history
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


if __name__ == "__main__":
    main()
