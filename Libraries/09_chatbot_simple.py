"""
09_chatbot_simple.py

Minimal Streamlit chatbot that retrieves context from lancedb and
asks GROQ (if configured) for an answer. Falls back to showing retrieved
passages when GROQ is not configured.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def get_table(path: str = "embedding_db"):
    import lancedb

    db = lancedb.connect(path)
    return db.open_table("chunks")


def ask_groq(question: str, context: str) -> str:
    try:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Model not configured. Set GROQ_API_KEY in .env"

        client = Groq(api_key=api_key)
        prompt = f"Answer using the context only.\nContext:\n{context}\nQuestion:\n{question}\nAnswer:"

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250,
        )

        return resp.choices[0].message.content
    except Exception as e:
        return f"Model Error: {e}"


st.set_page_config(page_title="Simple Doc Chat", layout="wide")
st.title("Simple Document Chatbot")

table = get_table()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the indexed documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # compute query embedding and search
    from lancedb.embeddings import get_registry

    embedder = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")
    qvec = embedder.compute_query_embeddings([prompt])[0]

    results = table.search(qvec).limit(5).to_pandas()
    texts = [row["text"] for _, row in results.iterrows()]
    context = "\n\n".join(texts)

    if not context:
        st.warning("No relevant content found in the index.")
    else:
        with st.chat_message("assistant"):
            answer = ask_groq(prompt, context)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
