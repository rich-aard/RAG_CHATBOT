from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")


def get_embeddings():
    return HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
