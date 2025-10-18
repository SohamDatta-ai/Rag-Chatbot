import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_api_key():
    print("=== API KEY TEST ===")
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API Key loaded: {'Yes' if api_key else 'No'}")
    print(f"Key starts with: {api_key[:10] if api_key else 'None'}")
    
    if not api_key:
        print("❌ No API key found in environment")
        return
    
    try:
        # Test with OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Test a simple completion
        print("\nTesting API key with simple request...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        
        print("✅ API key is working!")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        print("\nPossible issues:")
        print("- API key is invalid")
        print("- Account has no credits")
        print("- Account needs billing setup")
        print("- Rate limit exceeded")

if __name__ == "__main__":
    test_api_key()
