from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import streamlit as st
import os


@st.cache_resource(show_spinner="Loading PDF's...")
def load_pdfs(uploaded_pdfs):
    documents = []
    for file in uploaded_pdfs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            pdf_path = tmp.name
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
        finally:
            os.unlink(pdf_path)
    return documents


@st.cache_resource(show_spinner="Splitting documents...")
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents)


@st.cache_resource(show_spinner="Building vectorstore...")
def build_vectorstore(chunks_hash, chunks, google_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=google_api_key
    )
    return FAISS.from_documents(documents=chunks, embedding=embeddings)
