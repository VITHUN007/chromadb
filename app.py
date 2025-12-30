import os
import shutil
import tempfile
import streamlit as st
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
DB_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="PDF AI", layout="wide")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def clear_vdb():
    if "vectorstore" in st.session_state:
        st.session_state.vectorstore = None
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
            st.sidebar.success("Database wiped.")
        except PermissionError:
            st.sidebar.error("File locked! Close any applications using the database files and try again.")
    st.rerun()

st.title(" sample gemini Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        st.session_state.vectorstore = Chroma(
            persist_directory=DB_DIR, 
            embedding_function=load_embeddings()
        )
    else:
        st.session_state.vectorstore = None

with st.sidebar:
    if st.button("Reset Application"):
        clear_vdb()

if st.session_state.vectorstore is None:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None: 
        with st.spinner("Writing to database..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
                
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=load_embeddings(), 
                    persist_directory=DB_DIR
                )
                st.success("Indexing complete!")
                st.rerun()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path) 

else:
    st.info("Database Loaded")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    query = st.chat_input("Ask about the PDF...")
    
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        vdb = st.session_state.vectorstore    

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
        prompt = ChatPromptTemplate.from_template("""
            Answer based only on context:
            {context}
            Question: {input}
        """)
        
        chain = create_retrieval_chain(
            vdb.as_retriever(), 
            create_stuff_documents_chain(llm, prompt)
        )
        response = chain.invoke({"input": query})
        
        with st.chat_message("assistant"):
            st.markdown(response["answer"])

            docs_with_scores = vdb.similarity_search_with_score(query, k=1)
            st.subheader("Source Scores")
            for doc, score in docs_with_scores:
                with st.expander(f"Chunk (Distance: {score:.4f})"):
                    st.write(doc.page_content)


# 0.0 - 0.6	High Similarity: The chunk is very likely relevant to your question.
# 0.7 - 1.2	Moderate Similarity: The chunk may be somewhat relevant to your question."
# 1.3+	Low Similarity: The chunk is likely irrelevant to your question.