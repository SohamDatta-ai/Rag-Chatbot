import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Paths
CHROMA_PATH = "chroma_db"

def setup_free_system():
    print("=== RAG QUERY SYSTEM (FREE + GROQ) ===")
    print()
    
    # Check API keys
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("[ERROR] GROQ_API_KEY not found or not set.")
        print("Please update .env with your Groq API key.")
        return None, None, None
    
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

    # System prompt
    prompt_template = """
You are a helpful AI assistant. Use the retrieved context below to answer the user query accurately and concisely.
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:
"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    print("[OK] System initialized successfully!")
    print(f"[INFO] Using Groq model: llama-3.1-8b-instant")
    print(f"[INFO] Using free embeddings: all-MiniLM-L6-v2")
    print()
    
    return qa, llm, retriever

def ask_question(qa, query):
    try:
        result = qa({"query": query})
        print("🧠 Answer:\n", result["result"])
        print("\n📚 Sources:")
        for doc in result["source_documents"]:
            print(" -", doc.metadata.get("source", "Unknown file"))
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")

if __name__ == "__main__":
    qa, llm, retriever = setup_free_system()
    
    if qa:
        print("Ask a question about your documents:")
        user_query = input("👉 ")
        ask_question(qa, user_query)
