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
from langchain_core.prompts import ChatPromptTemplate
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

        system_prompt=("You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question: "
    "\n\n"
    "{context}"
    "\n\n"
    "If you don't know the answer, just say that you don't know.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(
            vectorstore.as_retriever(),
            combine_docs_chain
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        st.success(" RAG system ready!")

    query = st.text_input("Ask a question from the PDF")

    if query:
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke({"input": query})
            st.subheader(" Answer")
            st.write(response["answer"])
            st.divider()
        st.subheader("Source Chunks & Similarity Scores")
        
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=1)

        for i, (doc, score) in enumerate(docs_with_scores):
            with st.expander(f"Source {i+1} | Distance: {score:.4f}"):
                st.write(f"**Content:** {doc.page_content}")
                st.json(doc.metadata) 
    
# 0.0 - 0.6	High Similarity: The chunk is very likely relevant to your question.
# 0.7 - 1.2	Moderate Similarity: The chunk may be somewhat relevant to your question."
# 1.3+	Low Similarity: The chunk is likely irrelevant to your question.