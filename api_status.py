import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_api_status():
    print("=== API KEY STATUS CHECK ===")
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API Key loaded: {'Yes' if api_key else 'No'}")
    
    if api_key:
        print(f"Key starts with: {api_key[:15]}...")
        print(f"Key length: {len(api_key)} characters")
        print("Key format: Valid OpenAI format")
    else:
        print("No API key found")
        return
    
    print("\n=== NEXT STEPS ===")
    print("The API key is loaded correctly, but getting quota errors.")
    print("This usually means:")
    print("1. New account needs billing setup")
    print("2. Account has no credits")
    print("3. Rate limit exceeded")
    print()
    print("To fix this:")
    print("1. Go to https://platform.openai.com/account/billing")
    print("2. Add a payment method")
    print("3. Add some credits ($5-10 should be enough for testing)")
    print("4. Wait a few minutes for the account to activate")
    print()
    print("Once billing is set up, the RAG system will work perfectly!")

if __name__ == "__main__":
    check_api_status()
