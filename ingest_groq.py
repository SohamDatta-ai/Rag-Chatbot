import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For embeddings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # For LLM

# Directories
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"


def ingest_documents():
    print("=== RAG DOCUMENT INGESTION WITH GROQ ===")
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

    # Create embeddings (using OpenAI - very cheap)
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not found. Need this for embeddings.")
        print("You can get a free OpenAI API key for embeddings only.")
        return

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
    )

    # Store in ChromaDB
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"[OK] Stored embeddings in {CHROMA_PATH}")
    print()
    print("=== SYSTEM READY ===")
    print("[OK] Document processing: Complete")
    print("[OK] Embeddings: Generated")
    print("[OK] ChromaDB: Ready")
    print("[INFO] LLM: Will use Groq for queries")


if __name__ == "__main__":
    ingest_documents()
