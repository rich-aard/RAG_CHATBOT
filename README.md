# RAG Chatbot — PDF Q&A with Chat History

A conversational AI chatbot that lets you upload PDF documents and query their content using **Retrieval-Augmented Generation (RAG)**. Maintains full conversation history so follow-up questions are understood in context.

Built with **LangChain**, **Streamlit**, **Groq (Llama 3.3 70B)**, and **ChromaDB**.

---

## Features

- Upload one or multiple PDF files
- Semantic search over document content using vector embeddings
- History-aware retrieval — rewrites ambiguous follow-up questions using chat context
- Persistent in-session chat memory per user session
- Fast LLM inference via Groq (Llama 3.3 70B)
- Local embeddings via Ollama (`nomic-embed-text`) — no embedding API costs

---

## Architecture

```
User Question
      │
      ▼
History-Aware Retriever  ◄──── Chat History
      │
      ▼
ChromaDB Vector Search   ◄──── PDF Chunks (Ollama Embeddings)
      │
      ▼
Stuff Documents Chain
      │
      ▼
Groq LLM (Llama 3.3 70B)
      │
      ▼
    Answer
```

**Key design decisions:**
- The *context prompt* rewrites the user's question as a standalone query before retrieval, so history-dependent questions resolve correctly
- Embeddings are generated locally via Ollama — avoids latency and cost of remote embedding APIs
- Each browser session gets a unique `session_id` so chat histories are isolated per user

---

## Project Structure

```
RAG_CHATBOT/
├── src/
│   ├── __init__.py
│   ├── chains.py         # RAG chain + conversational chain builders
│   ├── history.py        # Session-scoped chat history (Streamlit state)
│   ├── llms.py           # LLM (Groq) and embeddings (Ollama) setup
│   ├── prompts.py        # Contextualize + QA system prompts
│   └── vectorstore.py    # PDF loading, text splitting, ChromaDB indexing
├── app.py                # Streamlit entry point
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally
- A free [Groq API key](https://console.groq.com/)

### 1. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### 2. Clone and install

```bash
git clone https://github.com/rich-aard/RAG_CHATBOT.git
cd rag-chatbot
```

Using **uv** (recommended):
```bash
uv sync
```

Using **pip**:
```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser and enter your **Groq API key** directly in the input field.

---

## Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| UI Framework     | [Streamlit](https://streamlit.io/)  |
| LLM              | Groq — Llama 3.3 70B Versatile      |
| Embeddings       | Ollama — nomic-embed-text (local)   |
| Vector Store     | [ChromaDB](https://www.trychroma.com/) |
| RAG Framework    | [LangChain](https://www.langchain.com/) + langchain-classic |
| PDF Parsing      | LangChain PyPDFLoader               |
| Package Manager  | [uv](https://github.com/astral-sh/uv) |

---

## Limitations

- Only text-based PDFs are supported — scanned/image-only PDFs will return no content
- Chat history is in-memory and resets on page refresh
- Answers are strictly grounded in the uploaded documents — the LLM will not use outside knowledge

---

