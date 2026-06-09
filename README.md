# RAG Chatbot — PDF Q&A with Chat History

A conversational AI chatbot that lets you upload PDF documents and query their content using **Retrieval-Augmented Generation (RAG)**. Maintains full conversation history so follow-up questions are understood in context.

Built with **LangChain**, **Streamlit**, **Groq (Llama 3.3 70B)**, **HuggingFace Embeddings**, and **FAISS**.

🚀 **Live Demo:** [ragchatbot-232.streamlit.app](https://ragchatbot-232.streamlit.app/)

---

## Features

- Upload one or multiple PDF files
- Semantic search over document content using vector embeddings
- History-aware retrieval — rewrites ambiguous follow-up questions using chat context
- Persistent in-session chat memory per user session
- Fast LLM inference via Groq (Llama 3.3 70B)
- Free embeddings via HuggingFace (`all-MiniLM-L6-v2`) — no embedding API costs
- In-memory FAISS vector store — no database setup required

---

## Architecture

```
User Question
      │
      ▼
History-Aware Retriever  ◄──── Chat History
      │
      ▼
FAISS Vector Search      ◄──── PDF Chunks (HuggingFace Embeddings)
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
- Embeddings are generated via HuggingFace `all-MiniLM-L6-v2` — downloaded at runtime, no API key required
- FAISS runs fully in-memory — no persistence layer, works cleanly on ephemeral deployments
- Each browser session gets a unique `session_id` so chat histories are isolated per user

---

## Project Structure

```
RAG_CHATBOT/
├── .streamlit/
│   └── config.toml       # Streamlit theme and server config
├── src/
│   ├── __init__.py
│   ├── chains.py         # RAG chain + conversational chain builders
│   ├── history.py        # Session-scoped chat history (Streamlit state)
│   ├── llms.py           # LLM (Groq) and embeddings (HuggingFace) setup
│   ├── prompts.py        # Contextualize + QA system prompts
│   └── vectorstore.py    # PDF loading, text splitting, FAISS indexing
├── app.py                # Streamlit entry point
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com/)

### 1. Clone and install

```bash
git clone https://github.com/rich-aard/RAG_CHATBOT.git
cd RAG_CHATBOT
```

Using **uv** (recommended):
```bash
uv sync
```

Using **pip**:
```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser and enter your **Groq API key** in the input field.

---

## Deployment (Streamlit Community Cloud)

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch, and set the main file path to `app.py`
4. Under **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "your-groq-key-here"
```

5. Hit **Deploy** — Streamlit Cloud handles the rest

> The HuggingFace model is downloaded automatically on first run. No additional API keys needed.

---

## Stack

| Component        | Technology                                                                 |
|------------------|----------------------------------------------------------------------------|
| UI Framework     | [Streamlit](https://streamlit.io/)                                         |
| LLM              | Groq — Llama 3.3 70B Versatile                                             |
| Embeddings       | HuggingFace — `sentence-transformers/all-MiniLM-L6-v2`                     |
| Vector Store     | [FAISS](https://github.com/facebookresearch/faiss) (in-memory)             |
| RAG Framework    | [LangChain](https://www.langchain.com/) + langchain-classic                |
| PDF Parsing      | LangChain PyPDFLoader                                                      |
| Package Manager  | [uv](https://github.com/astral-sh/uv)                                      |

---

## Limitations

- Only text-based PDFs are supported — scanned/image-only PDFs will return no content
- Chat history is in-memory and resets on page refresh
- Answers are strictly grounded in the uploaded documents — the LLM will not use outside knowledge

---