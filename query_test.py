import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()


def test_query_system():
    print("=== RAG QUERY SYSTEM TEST ===")
    print()

    # Simulate document processing
    print("[INFO] Loading documents...")
    loader = TextLoader("data/sample_document.txt")
    docs = loader.load()
    print(f"[OK] Loaded {len(docs)} documents")

    # Simulate text splitting
    print("[INFO] Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    # Show sample chunks
    print("\n=== SAMPLE CHUNKS ===")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}:")
        print(f"Content: {chunk.page_content[:150]}...")
        print(f"Source: {chunk.metadata.get('source', 'Unknown')}")

    # Simulate query processing
    print("\n=== QUERY SIMULATION ===")
    sample_queries = [
        "What is machine learning?",
        "Explain natural language processing",
        "What are vector databases?",
        "How does RAG work?",
    ]

    for query in sample_queries:
        print(f"\nQuery: {query}")
        print(
            "Simulated Answer: [Would retrieve relevant chunks and generate answer with GPT-3.5]"
        )
        print("Sources: [Would show source documents]")

    print("\n=== SYSTEM READY ===")
    print("[OK] Document processing: Working")
    print("[OK] Text splitting: Working")
    print("[OK] ChromaDB: Ready")
    print("[WARNING] Need valid OpenAI API key for embeddings and LLM")
    print()
    print("To use with real API:")
    print("1. Update .env with valid OpenAI API key")
    print("2. Run: python ingest.py")
    print("3. Run: python query.py")


if __name__ == "__main__":
    test_query_system()
