# app.py

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# --- UI ---
st.set_page_config(page_title="MOSDAC Chatbot", page_icon="🛰️")
st.title("🛰️ MOSDAC Chatbot (Local Version)")
st.markdown("Ask questions about satellite data, rainfall, or ocean observations from your uploaded documents.")

# --- Input ---
query = st.chat_input("Type your question here...")

if query:
    with st.spinner("Processing your query..."):
        try:
            # Load FAISS DB
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_type="similarity", k=3)

            # Connect to Ollama LLM (Mistral must be running)
            llm = Ollama(model="mistral")

            # Create QA chain
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            result = qa(query)

            # --- Output ---
            st.markdown(f"### ✅ Answer:\n{result['result']}")

            # --- Sources ---
            with st.expander("📄 Sources used"):
                for doc in result["source_documents"]:
                    st.markdown(f"- **{doc.metadata.get('source', 'Unknown')}**")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Make sure `ollama run mistral` is active in another terminal.")
