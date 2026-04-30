from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
import streamlit as st 

@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")