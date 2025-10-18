import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For embeddings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")     # For LLM

# Paths
CHROMA_PATH = "chroma_db"

def setup_groq_system():
    print("=== RAG QUERY SYSTEM WITH GROQ ===")
    print()
    
    # Check API keys
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not found. Need this for embeddings.")
        return None, None, None
    
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("[ERROR] GROQ_API_KEY not found or not set.")
        print("Please update .env with your Groq API key.")
        return None, None, None
    
    # Initialize embeddings and database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
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
    print(f"[INFO] Using OpenAI embeddings: text-embedding-3-small")
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
    qa, llm, retriever = setup_groq_system()
    
    if qa:
        print("Ask a question about your documents:")
        user_query = input("👉 ")
        ask_question(qa, user_query)
