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
   # RAG Chatbot (PDF + Streamlit + GPT-3.5)



# RAG Chatbot (PDF + Streamlit + GPT-3.5)

A local Retrieval-Augmented Generation (RAG) demo that lets you upload PDFs, index them, and ask questions that are answered using document context. Built for experimentation and local use.

## Features

- Upload PDF files and automatically index them
- Chunking + embeddings (sentence-transformers MiniLM)
- Local ChromaDB vector store (persisted to `chroma_db/`)
- Retrieval + optional GPT-3.5 (conversational answers)
- Simple Streamlit UI for quick interactions

## Requirements

- Python 3.11 (recommended) or Docker
- An OpenAI API key if you want GPT-3.5 responses

## Quick start (local)

1. Create a venv and activate it (PowerShell):

```powershell
python3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Create a `.env` with your OpenAI key (optional):

```
OPENAI_API_KEY=sk-...
```

3. Run the app:

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Quickstart (Docker)

If you prefer containerized runs:

```bash
docker compose build --pull
docker compose up
```

## Configuration

- `CHROMA_DIR` defaults to `chroma_db/` and is persisted.
- Embed model: `sentence-transformers/all-MiniLM-L6-v2` (downloaded on first use).
- Use `.env` to set `OPENAI_API_KEY` and other env vars.

## Development

- Format: `black` and `isort` are used (see `pyproject.toml`).
- Tests: run `python -m pytest -q`.
- CI: GitHub Actions workflow (runs tests on Python 3.11) is in `.github/workflows/ci.yml`.

## Tests

```bash
python -m pip install -r dev-requirements.txt
python -m pytest -q
```

## Troubleshooting

- If you hit NumPy / torch build errors on Windows, switch to Python 3.11 or use Docker.
- If model downloads hang, use the sidebar pre-download button in the app UI or install `sentence-transformers` manually.

## Contributing

Small pull requests are welcome — please keep changes focused (tests + formatting). Add a GH issue for larger features.

## License

MIT — see `LICENSE`.

## Contact

Built by Soham Datta — https://github.com/SohamDatta-ai

   > A lightweight RAG chatbot that lets you upload PDFs and chat with them — powered by LangChain, ChromaDB, and GPT-3.5.
   >
   > No setup drama. No cloud fees. Just your data + local brainpower.
>>>>>>> 6603bbc (chore: clean repo, refactor Streamlit entrypoint, add CI, license, formatting)
