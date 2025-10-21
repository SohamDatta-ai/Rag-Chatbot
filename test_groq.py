import os

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()


def test_groq_api():
    print("=== GROQ API TEST ===")

    # Get API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    print(f"Groq API Key loaded: {'Yes' if groq_api_key else 'No'}")

    if not groq_api_key:
        print("❌ No Groq API key found")
        return

    try:
        # Test with Groq client
        client = Groq(api_key=groq_api_key)

        # Test a simple completion
        print("\nTesting Groq API with simple request...")
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "user",
                    "content": "Hello, this is a test. Please respond with just 'Test successful'.",
                }
            ],
            max_tokens=10,
            temperature=0.1,
        )

        print("✅ Groq API is working!")
        print(f"Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"❌ Groq API test failed: {e}")


if __name__ == "__main__":
    test_groq_api()
