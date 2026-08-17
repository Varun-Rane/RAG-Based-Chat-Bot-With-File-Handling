import os
import tempfile
from typing import List
from pathlib import Path

import streamlit as st
import lancedb
import pandas as pd
from dotenv import load_dotenv
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="📁 Document Manager",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .doc-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD ENV
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# DATABASE
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
# PYDANTIC MODELS
# ---------------------------------------------------
class ChunkMetaData(LanceModel):
    filename: str | None = None
    page_numbers: List[int] | None = None
    title: str | None = None

class ChunkData(LanceModel):
    text: str
    embedding: Vector(get_embedder().ndims())  # type: ignore
    metadata: ChunkMetaData

# ---------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------
def extract_page_numbers(chunk) -> List[int] | None:
    """Extract page numbers from chunk metadata"""
    page_numbers = []

    for item in getattr(chunk.meta, "doc_items", []) or []:
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                page_numbers.append(page_no)

    unique_pages = sorted(set(page_numbers))
    return unique_pages or None

def process_document(file_path: str, file_name: str, progress_bar=None, status_text=None) -> bool:
    """Process and embed a single document"""
    try:
        if progress_bar:
            progress_bar.progress(10, "📥 Loading document...")
        if status_text:
            status_text.text("📥 Loading document...")

        converter = DocumentConverter()
        result = converter.convert(file_path)

        if progress_bar:
            progress_bar.progress(30, "✂️ Chunking document...")
        if status_text:
            status_text.text("✂️ Chunking document...")

        chunker = HybridChunker(max_tokens=500, merge_peers=True)
        chunks = list(chunker.chunk(dl_doc=result.document))

        if progress_bar:
            progress_bar.progress(50, "🔗 Computing embeddings...")
        if status_text:
            status_text.text("🔗 Computing embeddings...")

        db = init_db()
        embedder = get_embedder()

        # Check if table exists
        try:
            table = db.open_table("chunks")
            mode = "append"
        except Exception:
            table = db.create_table("chunks", schema=ChunkData, mode="overwrite")
            mode = "create"

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            headings = getattr(chunk.meta, "headings", None)

            metadata = ChunkMetaData(
                filename=file_name,
                page_numbers=extract_page_numbers(chunk),
                title=headings[0] if headings else None,
            )

            text = chunk.text
            embedding = embedder.compute_source_embeddings([text])[0]

            processed_chunks.append(ChunkData(
                text=text,
                embedding=embedding,
                metadata=metadata,
            ))

            # Update progress
            if progress_bar:
                progress = 50 + (i / len(chunks)) * 40
                progress_bar.progress(int(progress))

        if progress_bar:
            progress_bar.progress(90, "💾 Saving to database...")
        if status_text:
            status_text.text("💾 Saving to database...")

        if mode == "create":
            db.create_table("chunks", data=processed_chunks, schema=ChunkData, mode="overwrite")
        else:
            table.add(processed_chunks)

        if progress_bar:
            progress_bar.progress(100, "✅ Complete!")
        if status_text:
            status_text.text("✅ Complete!")

        return True

    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        return False

def get_document_stats():
    """Get statistics about documents in database"""
    db = init_db()
    try:
        table = db.open_table("chunks")
        df = table.search().to_pandas()

        if len(df) == 0:
            return {"total_docs": 0, "total_chunks": 0, "doc_list": []}

        # Extract unique documents
        doc_stats = {}
        for _, row in df.iterrows():
            filename = row["metadata"].get("filename", "Unknown")
            if filename not in doc_stats:
                doc_stats[filename] = {
                    "chunks": 0,
                    "pages": set(),
                    "titles": set()
                }
            doc_stats[filename]["chunks"] += 1

            pages = row["metadata"].get("page_numbers", [])
            if pages:
                doc_stats[filename]["pages"].update(pages)

            title = row["metadata"].get("title", "")
            if title:
                doc_stats[filename]["titles"].add(title)

        doc_list = []
        for filename, stats in doc_stats.items():
            doc_list.append({
                "Filename": filename,
                "Chunks": stats["chunks"],
                "Pages": len(stats["pages"]),
                "Titles": len(stats["titles"])
            })

        return {
            "total_docs": len(doc_stats),
            "total_chunks": len(df),
            "doc_list": doc_list
        }
    except Exception:
        return {"total_docs": 0, "total_chunks": 0, "doc_list": []}

# ---------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------

# Header
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("# 📁 Document Manager")
with col2:
    st.markdown("Manage your knowledge base: Upload, view, and delete documents")

st.divider()

# Main tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Document", "📊 Manage Documents", "📈 Statistics"])

# ---------------------------------------------------
# TAB 1: UPLOAD
# ---------------------------------------------------
with tab1:
    st.markdown("### Upload a New Document")
    st.info(
        "Supported formats:\n"
        "- 📄 PDF files\n"
        "- 🌐 URLs (web pages)\n"
        "- 📝 Text files"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Choose Upload Method:")
        upload_method = st.radio(
            "How would you like to upload?",
            ["📄 PDF File", "🌐 URL", "📝 Text Content"],
            label_visibility="collapsed",
            horizontal=False
        )

    st.divider()

    if upload_method == "📄 PDF File":
        st.markdown("#### PDF File Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            if st.button("🚀 Process PDF", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                success = process_document(tmp_path, uploaded_file.name, progress_bar, status_text)

                if success:
                    st.markdown("""
                    <div class='success-box'>
                    ✅ Document processed successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                    st.rerun()

                # Clean up
                Path(tmp_path).unlink()

    elif upload_method == "🌐 URL":
        st.markdown("#### URL Upload")
        url_input = st.text_input(
            "Enter URL:",
            placeholder="https://example.com/document.pdf",
            label_visibility="collapsed"
        )

        if url_input and st.button("🚀 Process URL", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            success = process_document(url_input, url_input.split("/")[-1], progress_bar, status_text)

            if success:
                st.markdown("""
                <div class='success-box'>
                ✅ URL processed successfully!
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1)
                st.rerun()

    else:  # Text Content
        st.markdown("#### Text Content Upload")
        text_content = st.text_area(
            "Paste your text here:",
            height=300,
            placeholder="Enter the content you want to embed...",
            label_visibility="collapsed"
        )

        if text_content:
            if st.button("🚀 Process Text", use_container_width=True):
                # Save text to temporary file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as tmp_file:
                    tmp_file.write(text_content)
                    tmp_path = tmp_file.name

                progress_bar = st.progress(0)
                status_text = st.empty()

                success = process_document(tmp_path, "text_content.txt", progress_bar, status_text)

                if success:
                    st.markdown("""
                    <div class='success-box'>
                    ✅ Text processed successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                    st.rerun()

                # Clean up
                Path(tmp_path).unlink()

# ---------------------------------------------------
# TAB 2: MANAGE
# ---------------------------------------------------
with tab2:
    st.markdown("### Manage Your Documents")

    stats = get_document_stats()

    if stats["total_docs"] == 0:
        st.info("📭 No documents in database yet. Upload your first document!")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📄 Total Documents", stats["total_docs"])
        with col2:
            st.metric("📦 Total Chunks", stats["total_chunks"])
        with col3:
            st.metric("📊 Avg Chunks/Doc", round(stats["total_chunks"] / stats["total_docs"], 1))

        st.divider()

        st.markdown("#### 📋 Documents List")

        if stats["doc_list"]:
            df = pd.DataFrame(stats["doc_list"])
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.divider()

            st.markdown("#### 🗑️ Delete Document")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                doc_to_delete = st.selectbox(
                    "Select document to delete:",
                    [doc["Filename"] for doc in stats["doc_list"]],
                    label_visibility="collapsed"
                )

            with col2:
                st.write("")

            with col3:
                if st.button("🗑️ Delete", use_container_width=True):
                    st.warning(f"Delete '{doc_to_delete}'? This cannot be undone.")

                    col_confirm1, col_confirm2 = st.columns(2)
                    with col_confirm1:
                        if st.button("✅ Confirm Delete", use_container_width=True):
                            db = init_db()
                            try:
                                table = db.open_table("chunks")
                                df = table.search().to_pandas()

                                # Filter out the document to delete
                                mask = df["metadata"].apply(lambda x: x.get("filename") != doc_to_delete)
                                df_filtered = df[mask]

                                # Recreate table without deleted document
                                embedder = get_embedder()
                                chunks_to_keep = []

                                for _, row in df_filtered.iterrows():
                                    metadata = ChunkMetaData(
                                        filename=row["metadata"].get("filename"),
                                        page_numbers=row["metadata"].get("page_numbers"),
                                        title=row["metadata"].get("title")
                                    )
                                    chunks_to_keep.append(ChunkData(
                                        text=row["text"],
                                        embedding=row["embedding"],
                                        metadata=metadata
                                    ))

                                if chunks_to_keep:
                                    db.create_table("chunks", data=chunks_to_keep, schema=ChunkData, mode="overwrite")
                                else:
                                    # Drop table if empty
                                    db.drop_table("chunks")

                                st.markdown("""
                                <div class='success-box'>
                                ✅ Document deleted successfully!
                                </div>
                                """, unsafe_allow_html=True)
                                time.sleep(1)
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error deleting document: {str(e)}")

                    with col_confirm2:
                        if st.button("❌ Cancel", use_container_width=True):
                            st.rerun()

# ---------------------------------------------------
# TAB 3: STATISTICS
# ---------------------------------------------------
with tab3:
    st.markdown("### 📈 Document Statistics")

    stats = get_document_stats()

    if stats["total_docs"] == 0:
        st.info("No data to display yet")
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Documents", stats["total_docs"], delta="📄")
        with col2:
            st.metric("Chunks", stats["total_chunks"], delta="📦")
        with col3:
            st.metric("Avg per Doc", round(stats["total_chunks"] / stats["total_docs"], 1), delta="📊")
        with col4:
            db = init_db()
            try:
                table = db.open_table("chunks")
                df = table.search().to_pandas()
                total_chars = sum(len(row["text"]) for _, row in df.iterrows())
                st.metric("Total Characters", f"{total_chars:,}", delta="📝")
            except:
                st.metric("Total Characters", "N/A", delta="📝")

        st.divider()

        st.markdown("#### Detailed Breakdown")

        if stats["doc_list"]:
            df = pd.DataFrame(stats["doc_list"])

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Chunks Distribution")
                chart_data = df.set_index("Filename")["Chunks"]
                st.bar_chart(chart_data)

            with col2:
                st.markdown("##### Pages Distribution")
                chart_data = df.set_index("Filename")["Pages"]
                st.bar_chart(chart_data)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9rem; padding: 2rem 0;'>
<p>📁 <strong>Document Manager</strong> | Manage your AI knowledge base</p>
</div>
""", unsafe_allow_html=True)
