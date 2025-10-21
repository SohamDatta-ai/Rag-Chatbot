import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Directories
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"


def ingest_documents():
    print("=== RAG DOCUMENT INGESTION (FREE EMBEDDINGS) ===")
    print()

    # Load documents (PDFs and text files)
    docs = []
    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
            print(f"[OK] Loaded PDF: {file}")
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
            docs.extend(loader.load())
            print(f"[OK] Loaded text: {file}")

    if not docs:
        print("[ERROR] No documents found in data folder!")
        return

    print(f"[INFO] Total documents loaded: {len(docs)}")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks")

    # Create embeddings using free HuggingFace model
    print("[INFO] Loading embedding model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("[OK] Embedding model loaded")

    # Store in ChromaDB
    print("[INFO] Creating embeddings and storing in ChromaDB...")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"[OK] Stored embeddings in {CHROMA_PATH}")
    print()
    print("=== SYSTEM READY ===")
    print("[OK] Document processing: Complete")
    print("[OK] Embeddings: Generated (free)")
    print("[OK] ChromaDB: Ready")
    print("[INFO] LLM: Will use Groq for queries")


if __name__ == "__main__":
    ingest_documents()
