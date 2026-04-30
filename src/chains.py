from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.prompts import context_prompt, qa_prompt
from src.history import get_session_history


def build_rag_chain(_llm, _retriever):
    history_aware_retriever = create_history_aware_retriever(
        _llm, _retriever, context_prompt
    )

    qna_chain = create_stuff_documents_chain(_llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qna_chain)

    return rag_chain


def build_conversational_chain(rag_chain):
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
