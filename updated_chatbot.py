import os
import tempfile
from typing import List

import lancedb
import streamlit as st
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from groq import Groq
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

# ---------------------------------------------------
# LOAD ENV
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------
embedder = get_registry().get("sentence-transformers").create(
    name="all-MiniLM-L6-v2"
)


class ChunkMetaData(LanceModel):
    filename: str | None = None
    page_numbers: List[int] | None = None
    title: str | None = None


class ChunkData(LanceModel):
    text: str
    embedding: Vector(embedder.ndims())  # type: ignore
    metadata: ChunkMetaData


# ---------------------------------------------------
# DATABASE
# ---------------------------------------------------
@st.cache_resource
def init_db():
    return lancedb.connect("embedding_db")


def get_table():
    db = init_db()
    try:
        return db.open_table("chunks")
    except Exception:
        return None


def extract_page_numbers(chunk) -> List[int] | None:
    page_numbers = []

    for item in getattr(chunk.meta, "doc_items", []) or []:
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                page_numbers.append(page_no)

    unique_pages = sorted(set(page_numbers))
    return unique_pages or None


def build_table_from_document(file_path: str, file_name: str):
    converter = DocumentConverter()
    result = converter.convert(file_path)

    chunker = HybridChunker(max_tokens=500, merge_peers=True)
    chunks = list(chunker.chunk(dl_doc=result.document))

    db = init_db()
    table = db.create_table("chunks", schema=ChunkData, mode="overwrite")

    processed_chunks = []

    for chunk in chunks:
        headings = getattr(chunk.meta, "headings", None)

        metadata = {
            "filename": getattr(getattr(chunk.meta, "origin", None), "filename", None)
            or file_name,
            "page_numbers": extract_page_numbers(chunk),
            "title": headings[0] if headings else None,
        }

        processed_chunks.append(
            {
                "text": chunk.text,
                "embedding": embedder.compute_source_embeddings([chunk.text])[0],
                "metadata": metadata,
            }
        )

    if processed_chunks:
        table.add(processed_chunks)

    return table, len(processed_chunks)


# ---------------------------------------------------
# GET CONTEXT FROM VECTOR SEARCH
# ---------------------------------------------------
def get_context(query: str, table, num_results: int = 5) -> str:
    try:
        query_vector = embedder.compute_query_embeddings([query])[0]

        results = table.search(query_vector).limit(num_results).to_pandas()

        context = []

        for _, row in results.iterrows():
            metadata = row["metadata"]

            filename = metadata.get("filename", "")
            page_numbers = metadata.get("page_numbers", [])
            title = metadata.get("title", "")

            source_parts = []

            if filename:
                source_parts.append(f"File: {filename}")

            if page_numbers:
                source_parts.append(f"Pages: {', '.join(str(p) for p in page_numbers)}")

            if title:
                source_parts.append(f"Title: {title}")

            source = " | ".join(source_parts)

            context.append(f"{row['text']}\nSource: {source}")

        return "\n\n".join(context)

    except Exception as e:
        st.error(f"Search Error: {str(e)}")
        return ""


# ---------------------------------------------------
# GET CHAT RESPONSE (GROQ)
# ---------------------------------------------------
def get_chat_response(question: str, context: str) -> str:
    try:
        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

Give a detailed, well-structured explanation in simple language.

Include:
1. Definition
2. How it works
3. Key features
4. Use cases
5. Important notes

If answer is not found in context, say:
"I could not find that in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Model Error: {str(e)}"


# ---------------------------------------------------
# UI CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Uploaded Document Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Uploaded Document Chatbot")
st.caption("Upload a document in the browser, then ask questions about its contents.")

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    st.session_state.active_document = None

if "table_ready" not in st.session_state:
    st.session_state.table_ready = False

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "pptx", "html", "md", "txt"],
    help="Choose a supported document format. The file will be indexed after upload.",
)

if uploaded_file is not None:
    should_reindex = (
        st.session_state.active_document != uploaded_file.name
        or not st.session_state.table_ready
    )

    if should_reindex:
        with st.status("Indexing uploaded document...", expanded=True) as status:
            status.write(f"Reading {uploaded_file.name}")

            suffix = os.path.splitext(uploaded_file.name)[1] or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            try:
                table, chunk_count = build_table_from_document(
                    temp_path, uploaded_file.name
                )
                st.session_state.active_document = uploaded_file.name
                st.session_state.table_ready = True
                st.session_state.table = table
                st.session_state.messages = []
                status.update(label="Document indexed", state="complete")
                st.success(f"Indexed {chunk_count} chunks from {uploaded_file.name}")
            except Exception as exc:
                st.session_state.table_ready = False
                st.session_state.active_document = None
                status.update(label="Indexing failed", state="error")
                st.error(f"Could not index document: {exc}")
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

# ---------------------------------------------------
# LOAD TABLE
# ---------------------------------------------------
table = st.session_state.get("table") or get_table()

if not table:
    st.info("Upload a document to start chatting.")
    st.stop()

# ---------------------------------------------------
# SHOW OLD CHAT
# ---------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.status("Searching document..."):
        context = get_context(prompt, table)

        if context:
            st.success("Relevant context found")

            with st.expander("View Retrieved Chunks"):
                st.text(context[:4000])
        else:
            st.warning("No relevant context found")

    with st.chat_message("assistant"):
        answer = get_chat_response(prompt, context)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})