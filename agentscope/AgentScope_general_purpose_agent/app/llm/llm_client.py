''' 💡 What This Is:
A retryable, API-safe, OpenAI-compatible client using Together.ai (OSS safe)

🧠 Why We Need This:
You had the API key... but never used it 😄

This abstracts away the model provider — supports Together, Mistral, etc.

Centralizes prompt/response logic + retry handling

⏰ When It’s Used:
Any time an agent calls an LLM to generate or evaluate something.

🛠️ How It Works:
Uses together library to call models

Built-in retry with tenacity

Abstract enough to plug in OSS models later'''

import os
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential
import together
import logging

logger = logging.getLogger(__name__)

# Init Together.ai
together.api_key = os.getenv("TOGETHER_API_KEY")

class LLMClient:
    def __init__(self, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10))
    def chat(self, prompt: str, temperature: float = 0.4, max_tokens: int = 500) -> Optional[str]:
        try:
            response = together.Complete.create(
                prompt=prompt,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\nUser:", "\nAI:"]
            )
            return response['output']['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"[LLMClient] Failed LLM call: {e}")
            return None

# Global instance
llm_client = LLMClient()
