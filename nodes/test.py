import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("API Key works!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"API Error: {e}")