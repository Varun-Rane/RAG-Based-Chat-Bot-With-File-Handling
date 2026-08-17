import streamlit as st

st.set_page_config(
    page_title="RAG Demo",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Assistant")
st.caption("Standalone RAG interface demo")

st.sidebar.header("Settings")
model = st.sidebar.selectbox(
    "Choose Model",
    ["Demo Model", "GPT Model", "Local Model"]
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx"]
)

question = st.text_input(
    "Ask a question",
    placeholder="Enter your question here..."
)

if uploaded_file:
    st.success("Document uploaded successfully!")

    st.write("Document:", uploaded_file.name)

    if question:
        st.subheader("Retrieved Context")

        context = (
            "This is sample retrieved context from "
            "your uploaded document."
        )

        st.info(context)

        st.subheader("Answer")

        answer = (
            f"Demo answer generated using {model}. "
            f"Your question was: {question}"
        )

        st.write(answer)

else:
    st.info("Upload a document to start.")

st.divider()

st.caption(
    "This is an isolated RAG UI demo. "
    "It does not connect to your existing application."
)