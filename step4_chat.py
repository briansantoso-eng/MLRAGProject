"""
Interactive RAG chat for the CloudDocs RAG system.

This module provides a conversational interface with history and
context-aware retrieval for follow-up questions.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import groq
import tiktoken
from config import (
    GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL, TEMPERATURE, MAX_TOKENS,
    CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_RETRIEVAL, MAX_CONVERSATION_HISTORY,
    STREAMING_ENABLED
)
import json

# Initialize clients
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
groq_client = groq.Groq(api_key=GROQ_API_KEY)

class RAGChat:
    """
    Interactive RAG chat with conversation memory.

    CONCEPT: Conversation memory allows follow-up questions to reference
    previous context. Query rewriting ensures the system understands
    pronouns like "it" or "that" in the context of prior questions.
    """

    def __init__(self, quiet=False):
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Conversation memory
        self.conversation_history = []

        if not quiet:
            print("🤖 RAG Chat initialized!")
            print("💡 Ask questions about AWS and Azure cloud services.")
            print("🔄 Type 'quit' to exit, 'clear' to reset memory.\n")

    def rewrite_query(self, user_query):
        """
        Rewrite follow-up questions to be self-contained.

        CONCEPT: Query rewriting adds context from conversation history
        so the retrieval system can find relevant information even for
        vague follow-up questions like "how does it compare?"
        """
        if not self.conversation_history:
            return user_query

        # If this seems like a follow-up question, add context
        follow_up_indicators = ["it", "that", "this", "those", "these", "them", "they"]

        query_lower = user_query.lower()
        if any(indicator in query_lower.split() for indicator in follow_up_indicators):
            # Get recent context
            recent_messages = self.conversation_history[-4:]  # Last 2 exchanges

            context_parts = []
            for msg in recent_messages:
                if msg["role"] == "user":
                    context_parts.append(f"User asked: {msg['content']}")
                elif msg["role"] == "assistant":
                    context_parts.append(f"Assistant said: {msg['content'][:200]}...")

            context = " ".join(context_parts)

            rewritten_query = f"Context: {context}\n\nCurrent question: {user_query}"
            return rewritten_query

        return user_query

    def retrieve_context(self, query, provider_filter=None):
        """
        Retrieve relevant context for the query.
        """
        # Get query embedding using local model
        query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()

        # Build filter
        where_clause = None
        if provider_filter:
            where_clause = {"provider": provider_filter}

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RETRIEVAL,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        context_chunks = []
        if results and results["documents"]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                similarity = 1 - distance
                context_chunks.append({
                    "text": doc,
                    "metadata": metadata,
                    "similarity": similarity
                })

        return context_chunks

    def build_chat_prompt(self, user_query, context_chunks):
        """
        Build prompt for chat response.
        """
        # Add context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"""
[Source {i}] {chunk['metadata']['title']} ({chunk['metadata']['provider'].upper()})
Similarity: {chunk['similarity']:.3f}
{chunk['text']}
""")

        context = "\n".join(context_parts)

        # Add conversation history
        history_parts = []
        for msg in self.conversation_history[-MAX_CONVERSATION_HISTORY*2:]:  # Last N exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {msg['content']}")

        history = "\n".join(history_parts)

        prompt = f"""You are a helpful cloud computing expert helping users understand AWS and Azure services.

CONVERSATION HISTORY:
{history}

RELEVANT DOCUMENTATION:
{context}

CURRENT QUESTION: {user_query}

INSTRUCTIONS:
- Answer based on the provided documentation and conversation history
- Be conversational but informative
- Reference specific services and their providers when relevant
- If you need more information, ask clarifying questions
- Keep responses focused and actionable

RESPONSE:"""

        return prompt

    def add_to_history(self, role, content):
        """
        Add message to conversation history.
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })

        # Trim history if too long
        if len(self.conversation_history) > MAX_CONVERSATION_HISTORY * 2:
            self.conversation_history = self.conversation_history[-MAX_CONVERSATION_HISTORY*2:]

    def generate_response(self, prompt, return_only=False):
        """
        Generate response using Groq (fast inference).
        """
        try:
            if not return_only and STREAMING_ENABLED:
                stream = groq_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=True
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
                response = groq_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
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

    def chat(self, user_query, provider_filter=None):
        """
        Process a single chat message.
        """
        # Rewrite query for context
        rewritten_query = self.rewrite_query(user_query)

        # Retrieve context
        context_chunks = self.retrieve_context(rewritten_query, provider_filter)

        if not context_chunks:
            response = "I don't have enough information in my knowledge base to answer that question. Could you try rephrasing it or asking about AWS/Azure cloud services?"
            print(f"🤖 {response}")
            self.add_to_history("user", user_query)
            self.add_to_history("assistant", response)
            return

        # Build prompt
        prompt = self.build_chat_prompt(user_query, context_chunks)

        # Show retrieved sources
        sources = [f"{chunk['metadata']['title']} ({chunk['metadata']['provider'].upper()})" for chunk in context_chunks[:3]]
        print(f"📚 Sources: {', '.join(sources)}")

        # Generate response
        print("🤖 ", end="")
        response = self.generate_response(prompt)

        # Add to history
        self.add_to_history("user", user_query)
        self.add_to_history("assistant", response)

    def run_interactive_chat(self):
        """
        Run the interactive chat loop.
        """
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Happy cloud computing!")
                    break

                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🧹 Conversation memory cleared.")
                    continue

                if user_input.lower().startswith('filter '):
                    # Handle provider filtering
                    parts = user_input.split(' ', 2)
                    if len(parts) >= 3:
                        provider = parts[1].lower()
                        if provider in ['aws', 'azure']:
                            query = parts[2]
                            print(f"🔍 Filtering to {provider.upper()} only...")
                            self.chat(query, provider)
                            continue

                # Normal chat
                self.chat(user_input)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

    def get_response(self, user_query, provider_filter=None):
        """
        Get response for web interface (returns string instead of printing).
        """
        # Rewrite query for context
        rewritten_query = self.rewrite_query(user_query)

        # Retrieve context
        context_chunks = self.retrieve_context(rewritten_query, provider_filter)

        if not context_chunks:
            response = "I don't have enough information in my knowledge base to answer that question. Could you try rephrasing it or asking about AWS/Azure/GCP cloud services?"
            self.add_to_history("user", user_query)
            self.add_to_history("assistant", response)
            return response

        # Build prompt
        prompt = self.build_chat_prompt(user_query, context_chunks)

        # Generate response
        response = self.generate_response(prompt, return_only=True)

        # Add source information
        if context_chunks:
            sources = []
            for chunk in context_chunks[:3]:  # Show top 3 sources
                metadata = chunk['metadata']
                provider = metadata.get('provider', 'unknown').upper()
                title = metadata.get('title', 'Unknown')
                url = metadata.get('url', '#')
                sources.append(f"- [{title}]({url}) ({provider})")

            if sources:
                response += "\n\n**📚 Sources:**\n" + "\n".join(sources)

        # Add to history
        self.add_to_history("user", user_query)
        self.add_to_history("assistant", response)

        return response

    