import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directories
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def ingest_documents_demo():
    print("=== RAG DOCUMENT INGESTION DEMO ===")
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
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks")
    
    # Show sample chunks
    print("\n=== SAMPLE CHUNKS ===")
    for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
        print(f"\nChunk {i+1}:")
        print(f"Content: {chunk.page_content[:200]}...")
        print(f"Metadata: {chunk.metadata}")
    
    print(f"\n[SUCCESS] Document processing complete!")
    print(f"[INFO] Ready for embedding generation with valid OpenAI API key")
    print(f"[INFO] ChromaDB directory: {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_documents_demo()
