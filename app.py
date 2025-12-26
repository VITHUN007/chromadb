import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub
load_dotenv()


st.set_page_config(page_title="sample app", layout="wide")
st.title("answer questions from PDF with Gemini")
st.write("Upload any PDF and ask questions from it")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path, extract_images=True)
        docs = loader.load()

        if len(docs) == 0:
            st.error(" No text found in the PDF")
            st.stop()

        st.success(f" Loaded {len(docs)} pages")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        st.write(f" Created {len(splits)} chunks")

        if len(splits) == 0:
            st.error("No chunks created")
            st.stop()

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

        prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(
            vectorstore.as_retriever(),
            combine_docs_chain
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        st.success(" RAG system ready!")

    query = st.text_input("Ask a question from the PDF")

    if query:
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke({"input": query})
            st.subheader(" Answer")
            st.write(response["answer"])
