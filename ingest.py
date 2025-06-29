# ingest.py

import os
from langchain_community.document_loaders import (PyPDFLoader,ReadTheDocsLoader,UnstructuredExcelLoader)
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_documents(path="data/"):
    docs = []
    failed_files = []

    for file in os.listdir(path):
        full_path = os.path.join(path, file)

        try:
            if file.endswith(".pdf"):
                print(f"📄 Loading PDF: {file}")
                docs.extend(PyPDFLoader(full_path).load())

            # elif file.endswith(".docx"):
            #     print(f"📝 Loading DOCX: {file}")
            #     docs.extend(UnstructuredWordDocumentLoader(full_path).load())
                
            # elif file.endswith(".docx"):
            #     print(f"📝 Loading DOCX: {file}")
            #     docs.extend(ReadTheDocsLoader(full_path).load())

            elif file.endswith(".xlsx"):
                print(f"📊 Loading XLSX: {file}")
                docs.extend(UnstructuredExcelLoader(full_path).load())

            else:
                print(f"⚠️ Skipped unsupported file: {file}")

        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
            failed_files.append((file, str(e)))
    
    return docs, failed_files

def build_vector_store():
    print("🔍 Loading documents...")
    docs, failed_files = load_documents()

    print(f"\n✅ Total documents loaded: {len(docs)}")
    if failed_files:
        print(f"\n❌ Failed to load {len(failed_files)} file(s):")
        for file, reason in failed_files:
            print(f" - {file} ❌ Reason: {reason}")

    if not docs:
        print("\n🚫 No valid documents found. Please check your data folder.")
        return

    print("\n✂️ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    print("\n🧠 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("💾 Saving FAISS vector store...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vector_store")
    print("✅ Vector store saved to 'vector_store/'")

if __name__ == "__main__":
    build_vector_store()
