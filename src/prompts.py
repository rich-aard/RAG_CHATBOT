from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


contextualize_system_prompt = """
    Given the chat history and the latest user question, rewrite the latest question as a standalone question that can be understood without the chat history.

    Rules:
    - Preserve the original meaning exactly
    - Replace ambiguous references with the specific subject from the chat history
    - Do not answer the question
    - Do not add new information or assumptions
    - If the question is already standalone, return it unchanged

    Return only the rewritten question.
    """

system_prompt = """
    You are a question-answering assistant.

    Answer the user's question using only the retrieved context provided below.

    Instructions:
    - If the answer is not explicitly supported by the context, say "I don't know based on the provided document."
    - Do not use outside knowledge
    - Be concise and accurate
    - Use at most 4 sentences
    - If the context contains partial information, state what is available without guessing

    Retrieved context:
    {context}
    """

context_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
