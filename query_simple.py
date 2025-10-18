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

def setup_simple_system():
    print("=== RAG QUERY SYSTEM (SIMPLE) ===")
    print()
    
    # Check API keys
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("[ERROR] GROQ_API_KEY not found or not set.")
        print("Please update .env with your Groq API key.")
        return None, None
    
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
    print(f"[INFO] Using Groq model: llama-3.1-8b-instant")
    print(f"[INFO] Using free embeddings: all-MiniLM-L6-v2")
    print()
    
    return llm, retriever

def ask_question_simple(llm, retriever, query):
    try:
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        
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
        
        print("Answer:\n", response.content)
        print("\nSources:")
        for doc in docs:
            print(" -", doc.metadata.get("source", "Unknown file"))
            
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")

if __name__ == "__main__":
    llm, retriever = setup_simple_system()
    
    if llm and retriever:
        print("Ask a question about your documents:")
        user_query = input("> ")
        ask_question_simple(llm, retriever, user_query)
