import streamlit as st
import hashlib
from src.chains import build_conversational_chain, build_rag_chain
from src.llms import get_embeddings, get_llm
from src.vectorstore import load_pdfs, build_vectorstore, split_documents
import uuid

# streamlit
st.title("[⌬RXZ.] Conversation RAG with PDF and chat history")
st.write("Upload PDF's and query their content.")

# api key
with st.sidebar:
    st.header("API KEY Configuration")
    
    groq_key = st.text_input(
            "Enter your GROQ API KEY: ", 
            type="password",
            )

if not groq_key:
    st.warning("Please enter your GROQ API key in the sidebar.")
    st.stop()


#models
llm = get_llm(api_key=groq_key)
embeddings = get_embeddings()

#session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

#file upload
uploaded_files = st.file_uploader(
    "Upload your PDF", type="pdf", accept_multiple_files=True
)
# process pdfs
if uploaded_files:
    #unique hash for uploaded files
    file_key = hashlib.md5(
        b"".join(f.getvalue() for f in uploaded_files)
    ).hexdigest()

    #doc processing
    documents = load_pdfs(uploaded_files)
    doc_split = split_documents(documents)
    vectorstore = build_vectorstore(chunks_hash=file_key, chunks=doc_split)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    rag_chain = build_rag_chain(_llm=llm, _retriever=retriever)
    conversation_rag_chain = build_conversational_chain(rag_chain=rag_chain)

    #prvious chat
    if "store" in st.session_state and session_id in st.session_state.store:
        for msg in st.session_state.store[session_id].messages:
            role = "user" if msg.type == "human" else "assistant"
            with st.chat_message(role):
                st.write(msg.content)

    #user input
    user_input = st.chat_input("Query: ")

    if user_input and user_input.strip():
        cleaned_input = user_input.strip()

        with st.chat_message("user"):
            st.markdown(cleaned_input)

        response = conversation_rag_chain.invoke(
            {"input": cleaned_input},
            config={"configurable": {"session_id": session_id}},
        )

        with st.chat_message("assistant"):
            st.markdown(response["answer"])

        with st.expander("Retrieved Sources"):
            for doc in response["context"]:
                st.write(f"Page: {doc.metadata.get('page', '0') + 1}")
                st.write(doc.page_content[:700])
                st.divider()