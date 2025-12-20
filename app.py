import streamlit as st
import chromadb
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

def get_vector_db(db_path="./my_notes_db"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name="personal_notes")

def run_manual_rag(user_query, _collection, _genai_client):
    
    results = _collection.query(query_texts=[user_query], n_results=1)
    
    if not results['documents'] or not results['documents'][0]:
        return "I couldn't find any relevant notes in your database."

    retrieved_context = results['documents'][0][0]

    prompt = f"""
    Answer the question based ONLY on the context below. 
    If you don't know, say "I don't have this in my notes."

    CONTEXT: {retrieved_context}
    QUESTION: {user_query}
    """
    try:
        response = _genai_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="My AI Notes", page_icon="📝")
    st.title("My Private Note Assistant")

    collection = get_vector_db()
    genai_client = genai.Client()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask me about your notes..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching your memory..."):
                answer = run_manual_rag(user_input, collection, genai_client)
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()