import streamlit as st
import hashlib
from src.chains import build_conversational_chain, build_rag_chain
from src.llms import get_embeddings, get_llm
from src.vectorstore import load_pdfs, build_vectorstore, split_documents
import uuid

# streamlit
st.title("[⌬RY] Conversation RAG with PDF and chat history")
st.write("Upload PDF's and query their content.")

# groq key
groq_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
google_key = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, "secrets") else ""

if not groq_key:
    groq_key = st.text_input("Enter your GROQ API KEY: ", type="password")
if not google_key:
    google_key = st.text_input("Enter your GOOGLE API KEY: ", type="password")

if groq_key and google_key:
    llm = get_llm(api_key=groq_key)
    embeddings = get_embeddings(google_api_key=google_key)

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    session_id = st.session_state.session_id

    uploaded_files = st.file_uploader(
        "Upload your PDF", type="pdf", accept_multiple_files=True
    )

    # process pdfs
    if uploaded_files:
        file_key = hashlib.md5(
            b"".join(f.getvalue() for f in uploaded_files)
        ).hexdigest()

        documents = load_pdfs(uploaded_files)
        doc_split = split_documents(documents)
        vectorstore = build_vectorstore(
            chunks_hash=file_key, chunks=doc_split, google_api_key=google_key
        )
        retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12}
        )

        rag_chain = build_rag_chain(_llm=llm, _retriever=retriever)

        conversation_rag_chain = build_conversational_chain(rag_chain=rag_chain)

        if "store" in st.session_state and session_id in st.session_state.store:
            for msg in st.session_state.store[session_id].messages:
                role = "user" if msg.type == "human" else "assistant"
                with st.chat_message(role):
                    st.write(msg.content)

        user_input = st.chat_input("Query: ")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            response = conversation_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            with st.expander("Retrieved Sources"):
                for doc in response["context"]:
                    st.write(f"Page: {doc.metadata.get('page', '0') + 1}")
                    st.write(doc.page_content[:700])
                    st.divider()

else:
    st.warning("Please enter both your GROQ and GOOGLE API keys.")
