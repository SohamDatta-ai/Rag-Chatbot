import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import CHROMA_DIR as UTILS_CHROMA_DIR
from utils import (
    has_been_ingested,
    load_ingested_index,
    mark_ingested,
    save_ingested_index,
    sha256_bytes,
)

load_dotenv()

CHROMA_DIR = UTILS_CHROMA_DIR
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_and_split_pdf(tmp_path: str):
    """Load PDF from path and split into Document chunks with metadata."""
    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    # Ensure each doc has a source metadata
    for d in docs:
        try:
            # prefer existing metadata 'source' if provided by loader
            if not d.metadata.get("source"):
                d.metadata["source"] = tmp_path
        except Exception:
            d.metadata = {"source": tmp_path}
    return docs


def ingest_pdf_bytes(file_bytes: bytes) -> tuple[int, str]:
    """Idempotent ingestion: returns (num_chunks, message).

    If a file with the same hash was already ingested, ingestion is skipped.
    """
    file_hash = sha256_bytes(file_bytes)
    if has_been_ingested(file_hash):
        return 0, "This PDF was already ingested. Skipping re-embedding."

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Load and split
        docs = load_and_split_pdf(tmp_path)

        # Embeddings (HuggingFace - free)
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create / update Chroma DB
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embed_model)

        # add_documents expects a list of Document objects
        db.add_documents(docs)
        db.persist()

        # mark ingested
        mark_ingested(file_hash, {"time": time.time(), "chunks": len(docs)})

        return len(docs), "PDF processed and added to database."
    finally:
        # Clean up the temporary file
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def get_db_and_retriever():
    """Lazy-load embedding model and Chroma DB. Returns (db, retriever) or (None, None)."""
    try:
        # If DB does not exist yet, return None; caller should instruct user to ingest first
        if not os.path.exists(CHROMA_DIR):
            return None, None

        # Cache embed model and db in session_state to avoid reloading
        if "embed_model" not in st.session_state:
            try:
                st.session_state.embed_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model: {e}\nTry running the pre-download button in the sidebar or install 'sentence-transformers' and 'torch'."
                )

        if "chroma_db" not in st.session_state:
            st.session_state.chroma_db = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=st.session_state.embed_model,
            )

        db = st.session_state.chroma_db
        retriever = db.as_retriever(search_kwargs={"k": 4})
        return db, retriever
    except Exception as e:
        st.error(f"Failed to initialize retrieval system: {e}")
        return None, None


def process_query(question, db, retriever, use_llm_flag):
    """Run either the conversational LLM chain (if enabled) or retrieval-only and return a dict with keys:
    - 'answer' (str) when LLM used
    - 'source_documents' (list) when LLM used
    - 'retrieved' (list of docs) when retrieval-only
    """
    # Ensure retriever/db available
    if retriever is None and db is None:
        return {"error": "No DB or retriever available"}

    # Attempt to use conversational chain when requested
    if use_llm_flag and OPENAI_API_KEY:
        # Ensure conv_chain exists
        if "conv_chain" not in st.session_state:
            try:
                from langchain.chains import ConversationalRetrievalChain
                from langchain.memory import ConversationBufferMemory
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
                memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                )
                st.session_state.conv_chain = ConversationalRetrievalChain.from_llm(
                    llm,
                    retriever=retriever,
                    memory=memory,
                    return_source_documents=True,
                )
            except Exception as e:
                return {"error": f"Failed to initialize conversational chain: {e}"}

        conv_chain = st.session_state.get("conv_chain")
        try:
            res = conv_chain({"question": question})
            answer = res.get("answer") or res.get("result") or res.get("output_text")
            source_docs = res.get("source_documents") or []
            return {"answer": answer, "source_documents": source_docs}
        except Exception as e:
            return {"error": f"LLM chain error: {e}"}

    # Retrieval-only fallback
    # Try retriever methods or DB-level similarity_search
    try:
        method = (
            getattr(retriever, "get_relevant_documents", None)
            or getattr(retriever, "retrieve", None)
            or getattr(retriever, "get_documents", None)
        )
        results = None
        if callable(method):
            results = method(question)
        else:
            if db is not None and hasattr(db, "similarity_search"):
                try:
                    results = db.similarity_search(question, k=4)
                except TypeError:
                    results = db.similarity_search(question)
            elif db is not None and hasattr(db, "similarity_search_with_score"):
                ss = db.similarity_search_with_score(question, k=4)
                results = [d for d, s in ss]

        if results is None:
            return {
                "error": "Retriever does not expose a retrieval method and DB fallback failed"
            }

        # normalize to list
        if not isinstance(results, (list, tuple)):
            results = list(results)

        return {"retrieved": results}
    except Exception as e:
        return {"error": f"Retrieval failed: {e}"}


def main():
    """Streamlit UI entrypoint."""
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("RAG Chatbot — Upload PDF & Auto-Ingest")

    with st.sidebar:
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        st.markdown("---")
        st.header("DB / Status")
        if st.button("Show DB files"):
            if os.path.exists(CHROMA_DIR):
                files = os.listdir(CHROMA_DIR)
                st.write(f"Chroma directory: {CHROMA_DIR}")
                st.write(files)
            else:
                st.warning("No Chroma DB found. Upload a PDF to create one.")

        st.markdown("---")
        st.header("LLM / Conversation")
        use_llm = st.checkbox(
            "Enable LLM answers (ChatGPT) — requires OPENAI_API_KEY in .env",
            value=False,
        )
        if use_llm and not OPENAI_API_KEY:
            st.warning(
                "OPENAI_API_KEY not found in environment. LLM disabled until key is provided."
            )
        if st.button("Pre-download embedding model"):
            with st.spinner("Downloading embedding model (this may take a while)..."):
                try:
                    _ = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    st.success("Embedding model is ready")
                except Exception as e:
                    st.error(f"Model download failed: {e}")
        st.markdown("---")
        st.header("Resume QA Quick Actions")
        if st.button("Summarize resume (3 sentences)"):
            st.session_state._quick_action = "summary"
        if st.button("Create 3 STAR stories"):
            st.session_state._quick_action = "star"
        st.write("Tailor to JD:")
        jd_text = st.text_area("Paste job description here (optional)")
        if st.button("Tailor bullets to JD"):
            st.session_state._quick_action = "tailor"
            st.session_state._jd_text = jd_text
        st.markdown("---")
        st.header("Chat history")
        if "chat_history" in st.session_state and st.session_state.chat_history:
            if st.button("Show chat history"):
                for item in st.session_state.chat_history:
                    st.write(f"Q: {item['question']}")
                    st.write(f"A: {item['answer']}")
        if st.button("Clear chat history"):
            st.session_state.chat_history = []

    ingest_info = None
    # uploaded_file and use_llm are defined in the sidebar block
    if uploaded_file is not None:
        with st.spinner(
            "Processing PDF and creating embeddings — this may take a while..."
        ):
            try:
                bytes_data = uploaded_file.read()
                count, _msg = ingest_pdf_bytes(bytes_data)
                ingest_info = f"PDF processed and {count} chunks added to Chroma DB."
                st.sidebar.success("PDF processed and added to database ✅")
            except Exception as e:
                st.sidebar.error(f"Ingestion failed: {e}")

    st.header("Chat (Retrieval-only for testing)")
    st.write(
        "This demo shows retrieval from the Chroma DB. An LLM can be added later to generate answers from retrieved chunks."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        # use session_state-backed text input so we can programmatically set the question for quick actions
        question = st.text_input(
            "Ask a question about your documents:", key="user_question"
        )

        # Determine whether to run the query: user pressed button OR a quick action was queued
        run_query = st.button("Retrieve") or (
            st.session_state.get("_quick_action") is not None
        )

        if run_query:
            # If quick action is queued and no question provided, set an appropriate question
            qa = st.session_state.get("_quick_action")
            if qa and not st.session_state.get("user_question"):
                if qa == "summary":
                    st.session_state.user_question = (
                        "Summarize the uploaded documents in 3 concise sentences."
                    )
                elif qa == "star":
                    st.session_state.user_question = "Create 3 STAR-format interview stories from the uploaded documents."
                elif qa == "tailor":
                    jd = st.session_state.get("_jd_text", "")
                    st.session_state.user_question = f"Tailor the resume to the following job description and suggest 3 rewritten bullets: {jd}"

            question = st.session_state.get("user_question")

            if not question:
                st.info("Enter a question to retrieve relevant chunks.")
            else:
                db, retriever = get_db_and_retriever()
                if retriever is None and db is None:
                    st.warning(
                        "No Chroma DB available. Upload a PDF first to create embeddings."
                    )
                else:
                    with st.spinner("Running query..."):
                        res = process_query(question, db, retriever, use_llm)

                    # Clear quick action after processing
                    if "_quick_action" in st.session_state:
                        del st.session_state["_quick_action"]
                    if "_jd_text" in st.session_state:
                        del st.session_state["_jd_text"]

                    # Display results or errors
                    if res.get("error"):
                        st.error(res["error"])
                    elif res.get("answer"):
                        st.subheader("Answer")
                        st.write(res.get("answer"))
                        srcs = res.get("source_documents", [])
                        if srcs:
                            st.subheader("Sources")
                            for d in srcs:
                                st.markdown(f"- {d.metadata.get('source', 'unknown')}")
                        # append to chat history
                        history = st.session_state.get("chat_history", [])
                        history.append(
                            {"question": question, "answer": res.get("answer")}
                        )
                        st.session_state.chat_history = history
                    elif res.get("retrieved"):
                        st.subheader("Retrieved chunks")
                        for i, doc in enumerate(res.get("retrieved")):
                            st.markdown(
                                f"**Chunk {i+1} — source:** {doc.metadata.get('source', 'unknown')}"
                            )
                            st.write(doc.page_content)

    with col2:
        st.markdown("---")
        st.write("Helpful tips:")
        st.write("- Upload a PDF via the sidebar to create a Chroma DB.")
        st.write(
            "- Retrieval here only shows matching chunks. Add an LLM later to convert them into fluent answers."
        )

    if "ingest_info" in locals() and ingest_info:
        st.success(ingest_info)


if __name__ == "__main__":
    main()
