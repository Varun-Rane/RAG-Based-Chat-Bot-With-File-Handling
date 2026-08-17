import os
import json
import time
from datetime import datetime
from typing import List, Dict
from pathlib import Path

import streamlit as st
import lancedb
from dotenv import load_dotenv
from groq import Groq
from lancedb.embeddings import get_registry
import pandas as pd
from io import BytesIO

# ---------------------------------------------------
# PAGE CONFIG & STYLING
# ---------------------------------------------------
st.set_page_config(
    page_title="🤖 AI Chatbot Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### AI Chatbot Pro\nAn intelligent document Q&A system powered by Groq & LanceDB"
    }
)

# Custom CSS for Beautiful UI
st.markdown("""
    <style>
    /* Main Theme */
    :root {
        --primary: #6366f1;
        --secondary: #ec4899;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chat messages styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        display: flex;
        gap: 1rem;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        justify-content: flex-end;
    }

    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }

    .source-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        text-align: center;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }

    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD ENV
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------
@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

client = get_groq_client()

# ---------------------------------------------------
# DATABASE INIT
# ---------------------------------------------------
@st.cache_resource
def init_db():
    return lancedb.connect("embedding_db")

@st.cache_resource
def get_embedder():
    return get_registry().get("sentence-transformers").create(
        name="all-MiniLM-L6-v2"
    )

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()

# ---------------------------------------------------
# DATABASE UTILITIES
# ---------------------------------------------------
def get_table():
    db = init_db()
    try:
        return db.open_table("chunks")
    except Exception:
        return None

def get_db_stats():
    """Get database statistics"""
    table = get_table()
    if table is None:
        return {"documents": 0, "chunks": 0, "size_mb": 0}

    try:
        df = table.search().to_pandas()
        unique_files = df["metadata"].apply(lambda x: x.get("filename", "Unknown")).nunique() if len(df) > 0 else 0
        size = len(df) if len(df) > 0 else 0
        return {
            "documents": unique_files,
            "chunks": size,
            "size_mb": round(size * 0.001, 2)
        }
    except Exception as e:
        st.error(f"Error getting stats: {str(e)}")
        return {"documents": 0, "chunks": 0, "size_mb": 0}

# ---------------------------------------------------
# SEARCH & RESPONSE
# ---------------------------------------------------
def search_context(query: str, num_results: int = 5, similarity_threshold: float = 0.3) -> tuple:
    """Search for relevant chunks"""
    table = get_table()
    if table is None:
        return "", []

    try:
        embedder = get_embedder()
        query_vector = embedder.compute_query_embeddings([query])[0]

        results = table.search(query_vector).limit(num_results).to_pandas()

        context_parts = []
        sources = []

        for _, row in results.iterrows():
            metadata = row["metadata"]
            filename = metadata.get("filename", "Unknown")
            pages = metadata.get("page_numbers", [])
            title = metadata.get("title", "")

            source_str = f"📄 {filename}"
            if pages:
                source_str += f" (Pages: {', '.join(map(str, pages))})"
            if title:
                source_str += f" | {title}"

            context_parts.append(f"{row['text']}")
            sources.append(source_str)

        context = "\n\n".join(context_parts)
        return context, sources

    except Exception as e:
        st.error(f"Search Error: {str(e)}")
        return "", []

def get_chat_response(question: str, context: str, mode: str = "qa") -> str:
    """Get response from Groq"""

    prompts = {
        "qa": f"""You are a helpful AI assistant answering questions based on provided documents.

CONTEXT:
{context}

QUESTION: {question}

Instructions:
- Answer ONLY using the provided context
- Be clear and structured
- Include key details and examples
- If the answer is not in the context, clearly state "I couldn't find this information in the documents"
- Keep response concise but complete

ANSWER:""",

        "summarize": f"""Summarize the following content in key points:

CONTENT:
{context}

Provide a structured summary with:
1. Main idea
2. Key points (3-5 bullets)
3. Conclusion

SUMMARY:""",

        "explain": f"""Explain the following in simple, easy-to-understand language:

CONTENT:
{context}

Instructions:
- Break down complex concepts
- Use examples where appropriate
- Keep explanations clear and concise

EXPLANATION:"""
    }

    prompt = prompts.get(mode, prompts["qa"])

    try:
        message = client.messages.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        return message.content[0].text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/static/img/streamlit_logo.svg",
             width=50)
with col2:
    st.markdown("""
    <div style='text-align: center;'>
    <h1 style='margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    🤖 AI Chatbot Pro
    </h1>
    <p style='margin: 0; color: #666;'>Intelligent Document Q&A System</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    pass

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings & Controls")

    # Mode Selection
    chat_mode = st.selectbox(
        "Chat Mode:",
        ["Q&A", "Summarize", "Explain"],
        help="Select the type of response you want"
    )

    mode_map = {"Q&A": "qa", "Summarize": "summarize", "Explain": "explain"}
    mode = mode_map[chat_mode]

    # Search Parameters
    st.markdown("#### 🔍 Search Settings")
    num_results = st.slider(
        "Number of results:",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant chunks to use for context"
    )

    similarity_threshold = st.slider(
        "Similarity threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum similarity score for results"
    )

    st.divider()

    # Database Stats
    st.markdown("#### 📊 Database Stats")
    stats = get_db_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
        <div class='stat-value'>{stats['documents']}</div>
        <div class='stat-label'>Documents</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='stat-card'>
        <div class='stat-value'>{stats['chunks']}</div>
        <div class='stat-label'>Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Utilities
    st.markdown("#### 🛠️ Utilities")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.conversation_history = []
        st.success("Conversation cleared!")
        st.rerun()

    if st.button("📥 Download Conversation"):
        if st.session_state.conversation:
            # Create markdown
            md_text = "# Conversation Export\n\n"
            md_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            for msg in st.session_state.conversation:
                md_text += f"**{msg['role'].upper()}**: {msg['content']}\n\n"

            st.download_button(
                label="📥 Download MD",
                data=md_text,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("No conversation to export yet")

    st.divider()

    # Info
    st.markdown("#### ℹ️ About")
    st.info(
        "**AI Chatbot Pro** uses:\n"
        "- 🔗 LanceDB for embeddings\n"
        "- 🧠 Groq API for LLM\n"
        "- 🤗 Sentence-Transformers for embeddings"
    )

# Main Content Area
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "📚 Documents"])

# TAB 1: CHAT
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Ask Questions About Your Documents")
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    # Chat Display
    chat_container = st.container()

    with chat_container:
        if len(st.session_state.conversation) == 0:
            st.markdown("""
            <div style='text-align: center; padding: 3rem; color: #999;'>
            <h3>👋 Welcome!</h3>
            <p>Start by asking a question about your documents below...</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.conversation:
                with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                    st.write(msg["content"])

                    if msg["role"] == "assistant" and "sources" in msg:
                        with st.expander("📌 Sources"):
                            for source in msg["sources"]:
                                st.markdown(f"<div class='source-box'>{source}</div>",
                                          unsafe_allow_html=True)

    st.divider()

    # Input Section
    st.markdown("### Your Question")

    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Ask anything about your documents:",
            placeholder="e.g., What is the main topic? How does X work?",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("📤 Send", use_container_width=True)

    # Process Query
    if send_button and user_input:
        if get_table() is None:
            st.error("❌ No documents in database. Please run the embedding pipeline first.")
        else:
            # Add user message
            st.session_state.conversation.append({
                "role": "user",
                "content": user_input
            })
            st.session_state.query_count += 1
            st.session_state.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": user_input,
                "mode": mode
            })

            # Get context and response
            with st.spinner("🔍 Searching and generating response..."):
                context, sources = search_context(user_input, num_results, similarity_threshold)

                if not context:
                    response = "I couldn't find any relevant information in the documents."
                    sources = ["No sources found"]
                else:
                    response = get_chat_response(user_input, context, mode)

            # Add bot response
            st.session_state.conversation.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })

            st.rerun()

# TAB 2: ANALYTICS
with tab2:
    st.markdown("### 📈 Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", st.session_state.query_count)

    with col2:
        stats = get_db_stats()
        st.metric("Total Documents", stats["documents"])

    with col3:
        stats = get_db_stats()
        st.metric("Total Chunks", stats["chunks"])

    with col4:
        session_duration = (datetime.now() - st.session_state.start_time).total_seconds() / 60
        st.metric("Session Duration (min)", f"{session_duration:.1f}")

    st.divider()

    # Query History
    if st.session_state.conversation_history:
        st.markdown("### 📋 Query History")

        history_data = []
        for item in st.session_state.conversation_history:
            history_data.append({
                "Time": item["timestamp"][:19],
                "Query": item["query"][:50] + "..." if len(item["query"]) > 50 else item["query"],
                "Mode": item["mode"].upper()
            })

        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        # Mode statistics
        st.markdown("### Query Mode Distribution")
        mode_counts = {}
        for item in st.session_state.conversation_history:
            mode = item["mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        if mode_counts:
            df_modes = pd.DataFrame(list(mode_counts.items()), columns=["Mode", "Count"])
            st.bar_chart(df_modes.set_index("Mode"))
    else:
        st.info("No query history yet. Start asking questions!")

# TAB 3: DOCUMENTS
with tab3:
    st.markdown("### 📚 Document Information")

    table = get_table()
    if table is None:
        st.warning("⚠️ No documents in database yet")
    else:
        try:
            df = table.search().to_pandas()

            if len(df) == 0:
                st.info("No chunks found")
            else:
                # Extract document info
                docs_info = []
                for _, row in df.iterrows():
                    metadata = row["metadata"]
                    docs_info.append({
                        "filename": metadata.get("filename", "Unknown"),
                        "pages": ", ".join(map(str, metadata.get("page_numbers", []))) if metadata.get("page_numbers") else "N/A",
                        "title": metadata.get("title", "N/A"),
                        "chunk_preview": row["text"][:100] + "..." if len(row["text"]) > 100 else row["text"]
                    })

                # Unique documents
                unique_docs = {}
                for doc in docs_info:
                    filename = doc["filename"]
                    if filename not in unique_docs:
                        unique_docs[filename] = {"count": 0, "pages": set()}
                    unique_docs[filename]["count"] += 1
                    if doc["pages"] != "N/A":
                        unique_docs[filename]["pages"].update(doc["pages"].split(", "))

                st.markdown("#### 📋 Documents Summary")
                for filename, info in unique_docs.items():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**📄 {filename}**")
                    with col2:
                        st.markdown(f"Chunks: `{info['count']}`")
                    with col3:
                        pages = len(info['pages']) if info['pages'] else 0
                        st.markdown(f"Pages: `{pages}`")

                st.divider()

                # Detailed chunk view
                st.markdown("#### 🔍 All Chunks")

                df_chunks = pd.DataFrame(docs_info)

                with st.expander("View all chunks"):
                    st.dataframe(
                        df_chunks[["filename", "title", "pages"]],
                        use_container_width=True,
                        hide_index=True
                    )

        except Exception as e:
            st.error(f"Error reading documents: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9rem; padding: 2rem 0;'>
<p>🚀 <strong>AI Chatbot Pro</strong> | Built with Streamlit, LanceDB & Groq</p>
<p>© 2024 | For questions and documents, use the chat interface above</p>
</div>
""", unsafe_allow_html=True)
