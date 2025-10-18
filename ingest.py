import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directories
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def ingest_documents():
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

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks")

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Store in ChromaDB
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"[OK] Stored embeddings in {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_documents()
