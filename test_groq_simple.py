import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

def test_groq_simple():
    print("=== GROQ SIMPLE TEST ===")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("No Groq API key found")
        return
    
    try:
        client = Groq(api_key=groq_api_key)
        
        # Test with llama-3.1-8b-instant (most common)
        print("Testing llama-3.1-8b-instant...")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            temperature=0.1
        )
        
        print("SUCCESS! Groq API is working")
        print("Response:", response.choices[0].message.content)
        print("Model: llama-3.1-8b-instant")
        
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_groq_simple()
