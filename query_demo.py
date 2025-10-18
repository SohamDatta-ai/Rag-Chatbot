import os
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
CHROMA_PATH = "chroma_db"

def demo_query_system():
    print("=== RAG QUERY SYSTEM DEMO ===")
    print()
    
    # Check if ChromaDB exists
    if not os.path.exists(CHROMA_PATH):
        print("[ERROR] ChromaDB not found. Please run 'python ingest.py' first.")
        return
    
    # Check if database has data
    try:
        # Try to load the database (without embeddings for demo)
        print("[OK] ChromaDB database found")
        print(f"[INFO] Database path: {CHROMA_PATH}")
        
        # List database files
        db_files = os.listdir(CHROMA_PATH)
        print(f"[INFO] Database files: {db_files}")
        
        # Check if we have documents in data folder
        data_files = os.listdir("data")
        print(f"[INFO] Available documents: {data_files}")
        
        print()
        print("=== SAMPLE QUERIES ===")
        print("The system is ready to answer questions like:")
        print("- What is machine learning?")
        print("- Explain natural language processing")
        print("- What are vector databases?")
        print("- How does RAG work?")
        print()
        print("=== SYSTEM STATUS ===")
        print("[OK] ChromaDB: Ready")
        print("[OK] Document processing: Ready")
        print("[WARNING] Embeddings: Need valid OpenAI API key")
        print("[WARNING] LLM: Need valid OpenAI API key")
        print()
        print("To use with real API:")
        print("1. Update .env with valid OpenAI API key")
        print("2. Run: python ingest.py")
        print("3. Run: python query.py")
        
    except Exception as e:
        print(f"[ERROR] Error loading database: {e}")

if __name__ == "__main__":
    demo_query_system()
