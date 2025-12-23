import os
import langchain
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub

# ----------------- CONFIG -----------------
st.set_page_config(page_title="PDF RAG App", layout="wide")
st.title("📄 Placement PDF Q&A App")

load_dotenv()

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    return embeddings, llm


# ----------------- LOAD & EMBED PDF -----------------
@st.cache_resource
def create_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path, extract_images=True)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=25,
        chunk_overlap=15
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore


embeddings, llm = load_models()

# ----------------- SIDEBAR -----------------
st.sidebar.header("📂 PDF Settings")

pdf_file = st.sidebar.file_uploader(
    "Upload Placement PDF",
    type=["pdf"]
)

if pdf_file:
    pdf_path = f"temp_{pdf_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    st.sidebar.success("PDF uploaded successfully!")

    vectorstore = create_vectorstore(pdf_path)

    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(
        vectorstore.as_retriever(),
        combine_docs_chain
    )

    # ----------------- CHAT UI -----------------
    st.subheader("💬 Ask Questions")

    user_question = st.text_input(
        "Enter your question",
        placeholder="What are the main points of this document?"
    )

    if st.button("Ask"):
        if user_question.strip():
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_question})
                st.markdown("### ✅ Answer")
                st.write(response["answer"])
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload a PDF to start asking questions.")
