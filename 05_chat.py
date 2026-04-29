import os
import streamlit as st
import lancedb
from dotenv import load_dotenv
from groq import Groq
from lancedb.embeddings import get_registry

# ---------------------------------------------------
# LOAD ENV
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------------------------------------------------
# DATABASE
# ---------------------------------------------------
@st.cache_resource
def init_db():
    db = lancedb.connect("embedding_db")
    table = db.open_table("chunks")
    return table

# ---------------------------------------------------
# GET CONTEXT FROM VECTOR SEARCH
# ---------------------------------------------------
def get_context(query: str, table, num_results: int = 5) -> str:
    try:
        func = get_registry().get("sentence-transformers").create(
            name="all-MiniLM-L6-v2"
        )

        query_vector = func.compute_query_embeddings([query])[0]

        results = (
            table.search(query_vector)
            .limit(num_results)
            .to_pandas()
        )

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
                source_parts.append(
                    f"Pages: {', '.join(str(p) for p in page_numbers)}"
                )

            if title:
                source_parts.append(f"Title: {title}")

            source = " | ".join(source_parts)

            context.append(
                f"{row['text']}\nSource: {source}"
            )

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
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Model Error: {str(e)}"

# ---------------------------------------------------
# UI CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Context-Aware Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Context-Aware Chatbot")

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# LOAD TABLE
# ---------------------------------------------------
table = init_db()

# ---------------------------------------------------
# SHOW OLD CHAT
# ---------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
if prompt := st.chat_input("Ask a question about the documents..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.status("Searching documents..."):

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

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )