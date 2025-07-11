# llm/generate.py

import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_response(prompt: str, model="llama3-8b-8192") -> str:
    if not GROQ_API_KEY:
        return "❌ No API key found. Please set GROQ_API_KEY."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=15
        )
        res.raise_for_status()
        data = res.json()

        if "choices" not in data:
            print("❌ Unexpected Groq API response:", data)
            return f"❌ Groq API error: {data.get('error', {}).get('message', 'No choices returned')}"

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        print("❌ Request failed:", e)
        return f"❌ Request error: {str(e)}"
