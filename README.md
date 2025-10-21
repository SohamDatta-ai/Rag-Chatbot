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

   > Upload a PDF. Ask it anything. Watch it answer like ChatGPT — but trained on your doc.

   Think of this as a tiny local ChatGPT for your PDFs. No cloud secrets (unless you opt in), no magic — just you, your docs, and a few open-source building blocks.

   ---

   ## 🧭 TL;DR (Hook)

   Upload a PDF. Ask it anything. Get grounded answers pulled straight from your document(s).

   ---

   ## ⚙️ What it does

   - 📄 Upload PDFs → automatically indexed
   - 💬 Chat with your docs using GPT-3.5 (optional)
   - 🔎 Instant answers grounded in your files (with sources)
   - 🧠 Keeps short-term conversation memory for follow-ups
   - 🛠️ Runs locally with Streamlit — no cloud required

   ---

   ## 🧩 How it works (simple breakdown)

   ```
   📄 PDF → 🔍 Text Splitter → 🧠 Embeddings → 💾 ChromaDB → 🤖 GPT-3.5 → 💬 Answer
   ```

   PDF is split into chunks. Each chunk gets an embedding (vector). Vectors are stored in ChromaDB. When you ask a question, the system finds the most relevant chunks and (optionally) asks GPT-3.5 to synthesize a grounded answer using those chunks as context.

   It’s quick, local, and intentionally minimal — perfect for testing ideas or getting instant answers from docs you actually control.

   ---

   ## 🚀 Quickstart (for devs)

   Try this — should take < 5 minutes if you already have Python and pip:

   ```bash
   git clone https://github.com/SohamDatta-ai/Rag-Chatbot.git
   cd Rag-Chatbot
   pip install -r requirements.txt
   python -m streamlit run app.py
   ```

   💡 Make sure you’ve set your `OPENAI_API_KEY` in a local `.env` if you want GPT-3.5 answers.

   ---

   ## 🧠 Stack

   - **LangChain** → orchestration
   - **Chroma** → vector DB
   - **HuggingFace MiniLM-L6-v2** → embeddings
   - **GPT-3.5-Turbo** → answers
   - **Streamlit** → UI

   ---

<<<<<<< HEAD


Beta Realease Coming soon

=======
   ## 🌍 Future upgrades (features I want to build)

   - 🔥 Multi-PDF chat (upload a folder)
   - 💾 Cloud vector DB (Pinecone / FAISS / Weaviate)
   - 🧠 Personal AI memory (opt-in long-term memory)
   - ☁️ One-click deploy (Heroku / Docker + GH Actions)

   ---

   ## 💬 Credits

   Built with ❤️ by [Soham Datta](https://github.com/SohamDatta-ai)

   ---

   ## ✅ Example output (tone preview)

   > **“Upload. Ask. Understand.”**
   >
   > A lightweight RAG chatbot that lets you upload PDFs and chat with them — powered by LangChain, ChromaDB, and GPT-3.5.
   >
   > No setup drama. No cloud fees. Just your data + local brainpower.
>>>>>>> 6603bbc (chore: clean repo, refactor Streamlit entrypoint, add CI, license, formatting)
