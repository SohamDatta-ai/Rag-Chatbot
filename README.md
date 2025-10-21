# RAG Chatbot Project

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, and Streamlit.

## Project Structure

```
rag-chatbot/
├── .venv/                 # Virtual environment
├── .env                   # Environment variables
├── .gitignore            # Git ignore file
├── requirements.txt         # Dependencies
├── data/                 # Document storage
│   └── sample_document.txt
├── chroma_db/            # Vector database
│   └── chroma.sqlite3
├── ingest.py             # Document ingestion script
├── ingest_demo.py        # Demo version (no API calls)
├── query.py              # Query system with LLM
├── query_demo.py         # Demo query system
├── query_test.py         # Test query system
├── hello_streamlit.py    # Test Streamlit app
└── README.md             # This file
```

## Setup

1. **Environment Setup:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Update `.env` with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   EMBEDDING_MODEL=text-embedding-3-small
   LLM_MODEL=gpt-3.5-turbo
   ```

## Usage

### Document Ingestion
```bash
# Process documents and create embeddings
python ingest.py

# Demo version (no API calls)
python ingest_demo.py
```

### Query System
```bash
# Interactive query system
python query.py

# Demo version
python query_demo.py

# Test version
python query_test.py
```

### Streamlit Interface
```bash
# Start web interface
.\.venv\Scripts\streamlit.exe run hello_streamlit.py
```

## Features

- ✅ Document loading (PDF + text files)
- ✅ Text chunking and splitting
- ✅ Vector embeddings with OpenAI
- ✅ ChromaDB vector storage
- ✅ Retrieval system
- ✅ LLM integration with GPT-3.5
- ✅ Source document citations
- ✅ Streamlit web interface

## System Status

- ✅ Virtual environment: Active
- ✅ All packages: Installed
- ✅ Document processing: Working
- ✅ ChromaDB: Ready
- ✅ Streamlit: Ready
- ⚠️ API integration: Needs valid OpenAI key



Beta Realease Coming soon

