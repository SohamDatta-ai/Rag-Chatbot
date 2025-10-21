import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
CHROMA_PATH = "chroma_db"

# Initialize embeddings and database
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Initialize retriever
retriever = db.as_retriever(search_kwargs={"k": 4})

# LLM initialization
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY)

# System prompt
prompt_template = """
You are a helpful AI assistant. Use the retrieved context below to answer the user query accurately and concisely.
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
)


def ask_question(query):
    result = qa({"query": query})
    print("🧠 Answer:\n", result["result"])
    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        print(" -", doc.metadata.get("source", "Unknown file"))


if __name__ == "__main__":
    print("Ask a question about your documents:")
    user_query = input("👉 ")
    ask_question(user_query)
