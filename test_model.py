import asyncio
import os
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

async def test_model():
    load_dotenv()
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )
    
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    print(f"Testing model: {model}")
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Translate to Spanish: Hello, how are you?"},
            ],
            max_completion_tokens=100,
        )
        print(f"Response: '{response.choices[0].message.content}'")
        print(f"Usage: {response.usage}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_model())
