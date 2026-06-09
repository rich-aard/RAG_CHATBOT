from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st


@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")


def get_embeddings(google_api_key: str, task_type: str = "retrieval_query"):
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key,
        task_type=task_type,
    )
