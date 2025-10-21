import os

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()


def check_groq_models():
    print("=== GROQ AVAILABLE MODELS ===")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("No Groq API key found")
        return

    try:
        client = Groq(api_key=groq_api_key)

        # Test with different models
        models_to_test = [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama-3.2-3b-preview",
            "llama-3.2-11b-preview",
            "llama-3.2-90b-preview",
            "mixtral-8x7b-32768",
        ]

        print("Testing available models...")

        for model in models_to_test:
            try:
                print(f"\nTesting {model}...")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                    temperature=0.1,
                )
                print(f"✅ {model} - Working!")
                print(f"Response: {response.choices[0].message.content}")
                break  # Use the first working model

            except Exception as e:
                print(f"❌ {model} - Failed: {str(e)[:100]}...")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_groq_models()
