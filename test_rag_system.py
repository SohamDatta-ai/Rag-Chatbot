import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Paths
CHROMA_PATH = "chroma_db"

def test_rag_system():
    print("=== RAG SYSTEM TEST ===")
    print()
    
    # Check API keys
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("[ERROR] GROQ_API_KEY not found or not set.")
        return
    
    # Initialize embeddings and database
    print("[INFO] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("[OK] Embedding model loaded")
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Initialize retriever
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # LLM initialization with Groq
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )
    
    print("[OK] System initialized successfully!")
    print()
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain natural language processing",
        "What are vector databases?",
        "How does RAG work?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        try:
            # Get relevant documents
            docs = retriever.invoke(query)
            
            # Create context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            prompt = f"""You are a helpful AI assistant. Use the retrieved context below to answer the user query accurately and concisely.
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""
            
            # Get response from Groq
            response = llm.invoke(prompt)
            
            print("Answer:", response.content)
            print("Sources:", [doc.metadata.get("source", "Unknown") for doc in docs])
            print("-" * 50)
            
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            print("-" * 50)

if __name__ == "__main__":
    test_rag_system()
