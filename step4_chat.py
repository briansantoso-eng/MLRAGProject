"""Interactive RAG chat with conversation memory and real-time streaming."""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import groq
from config import (
    GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL, TEMPERATURE, MAX_TOKENS,
    CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_RETRIEVAL,
    MAX_CONVERSATION_HISTORY, STREAMING_ENABLED,
)

# ── Module-level singletons ───────────────────────────────────────────────────

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
groq_client     = groq.Groq(api_key=GROQ_API_KEY)


class RAGChat:
    """
    Conversational RAG interface with memory and query rewriting.

    Two public entry points:
      chat()         — CLI use, streams response to stdout
      get_response() — Web UI use, returns response as a string
    """

    def __init__(self, quiet=False):
        # Connect to (or create) the ChromaDB collection on disk
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.conversation_history = []

        if not quiet:
            print("🤖 RAG Chat initialized!")
            print("💡 Ask questions about AWS, Azure, and GCP cloud services.")
            print("🔄 Type 'quit' to exit, 'clear' to reset memory.\n")

    # ── Query rewriting ───────────────────────────────────────────────────────

    def rewrite_query(self, user_query):
        """
        Expand follow-up questions with conversation context.
        Handles pronouns ('it', 'that', etc.) so the vector search stays accurate.
        """
        if not self.conversation_history:
            return user_query

        follow_up_words = {"it", "that", "this", "those", "these", "them", "they"}
        if not any(w in follow_up_words for w in user_query.lower().split()):
            return user_query

        # Prefix the query with a summary of the last 2 exchanges (4 messages)
        context = " ".join(
            f"User asked: {m['content']}" if m["role"] == "user"
            else f"Assistant said: {m['content'][:200]}..."
            for m in self.conversation_history[-4:]
        )
        return f"Context: {context}\n\nCurrent question: {user_query}"

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_context(self, query, provider_filter=None):
        """Embed query and return top-K matching chunks from ChromaDB."""
        query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RETRIEVAL,
            where={"provider": provider_filter} if provider_filter else None,
            include=["documents", "metadatas", "distances"],
        )
        if not results:
            return []
        if not results["documents"]:
            return []
        docs  = results["documents"][0]
        metas = (results["metadatas"] or [[]])[0]
        dists = (results["distances"] or [[]])[0]
        return [
            {"text": doc, "metadata": meta, "similarity": 1 - dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    # ── Prompt building ───────────────────────────────────────────────────────

    def build_chat_prompt(self, user_query, context_chunks):
        """Combine retrieved sources and conversation history into a single prompt."""
        context = "\n".join(
            f"[Source {i}] {c['metadata']['title']} ({c['metadata']['provider'].upper()})\n"
            f"Similarity: {c['similarity']:.3f}\n{c['text']}"
            for i, c in enumerate(context_chunks, 1)
        )
        history = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in self.conversation_history[-MAX_CONVERSATION_HISTORY * 2:]
        )
        return (
            f"You are a helpful cloud computing expert helping users understand AWS and Azure services.\n\n"
            f"CONVERSATION HISTORY:\n{history}\n\n"
            f"RELEVANT DOCUMENTATION:\n{context}\n\n"
            f"CURRENT QUESTION: {user_query}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Answer based on the provided documentation and conversation history\n"
            f"- Be conversational but informative\n"
            f"- Reference specific services and their providers when relevant\n"
            f"- If you need more information, ask clarifying questions\n"
            f"- Keep responses focused and actionable\n\n"
            f"RESPONSE:"
        )

    # ── History management ────────────────────────────────────────────────────

    def add_to_history(self, role, content):
        """Append a message and trim to MAX_CONVERSATION_HISTORY pairs."""
        self.conversation_history.append({"role": role, "content": content})
        max_msgs = MAX_CONVERSATION_HISTORY * 2
        if len(self.conversation_history) > max_msgs:
            self.conversation_history = self.conversation_history[-max_msgs:]

    # ── Generation ────────────────────────────────────────────────────────────

    def generate_response(self, prompt, return_only=False):
        """
        Call Groq LLM. Streams tokens to stdout when return_only=False.
        Uses Groq's native streaming API — first token appears in ~100–300ms.
        """
        try:
            if not return_only and STREAMING_ENABLED:
                # Stream tokens as they arrive — no artificial delay
                stream = groq_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=True,
                )
                response_text = ""
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        print(content, end="", flush=True)
                        response_text += content
                print()
            else:
                # Non-streaming path — used by get_response() for the web UI
                response      = groq_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = response.choices[0].message.content or ""
                if not return_only:
                    print(response_text)

            return response_text

        except Exception as e:
            error_msg = f"Error generating response with Groq API: {str(e)}"
            if not return_only:
                print(f"❌ {error_msg}")
            raise Exception(error_msg + "\nPlease check your GROQ_API_KEY is set correctly in Secrets.")

    # ── Public chat interfaces ────────────────────────────────────────────────

    def chat(self, user_query, provider_filter=None):
        """Process a CLI chat turn — retrieves context, streams response."""
        rewritten      = self.rewrite_query(user_query)
        context_chunks = self.retrieve_context(rewritten, provider_filter)

        if not context_chunks:
            response = (
                "I don't have enough information in my knowledge base to answer that question. "
                "Could you try rephrasing it or asking about AWS/Azure cloud services?"
            )
            print(f"🤖 {response}")
            self.add_to_history("user", user_query)
            self.add_to_history("assistant", response)
            return

        sources = ", ".join(
            f"{c['metadata']['title']} ({c['metadata']['provider'].upper()})"
            for c in context_chunks[:3]
        )
        print(f"📚 Sources: {sources}")
        print("🤖 ", end="")
        response = self.generate_response(self.build_chat_prompt(user_query, context_chunks))
        self.add_to_history("user", user_query)
        self.add_to_history("assistant", response)

    def get_response(self, user_query, provider_filter=None):
        """Return response as a string with source citations (used by Streamlit UI)."""
        rewritten      = self.rewrite_query(user_query)
        context_chunks = self.retrieve_context(rewritten, provider_filter)

        if not context_chunks:
            response = (
                "I don't have enough information in my knowledge base to answer that question. "
                "Could you try rephrasing it or asking about AWS/Azure/GCP cloud services?"
            )
            self.add_to_history("user", user_query)
            self.add_to_history("assistant", response)
            return response

        response = self.generate_response(
            self.build_chat_prompt(user_query, context_chunks),
            return_only=True,
        )

        # Append markdown source links for the web UI
        sources = [
            f"- [{c['metadata'].get('title', 'Unknown')}]({c['metadata'].get('url', '#')}) "
            f"({c['metadata'].get('provider', 'unknown').upper()})"
            for c in context_chunks[:3]
        ]
        if sources:
            response += "\n\n**📚 Sources:**\n" + "\n".join(sources)

        self.add_to_history("user", user_query)
        self.add_to_history("assistant", response)
        return response

    # ── Interactive CLI loop ──────────────────────────────────────────────────

    def run_interactive_chat(self):
        """Run the REPL-style CLI chat loop."""
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "bye"):
                    print("👋 Goodbye! Happy cloud computing!")
                    break
                if user_input.lower() == "clear":
                    self.conversation_history = []
                    print("🧹 Conversation memory cleared.")
                    continue
                # Optional syntax: "filter aws <question>"
                if user_input.lower().startswith("filter "):
                    parts = user_input.split(" ", 2)
                    if len(parts) >= 3 and parts[1].lower() in ("aws", "azure", "gcp"):
                        print(f"🔍 Filtering to {parts[1].upper()} only...")
                        self.chat(parts[2], parts[1].lower())
                        continue
                self.chat(user_input)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
