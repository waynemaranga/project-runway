import httpx
from typing import Optional, Any
from httpx import RequestError, Response
import dotenv
from pathlib import Path

ENV_PATH = str(Path(__file__).resolve().parent / ".env")
AZURE_EXTRAS_URI: Optional[str] = dotenv.get_key(ENV_PATH, "AZURE_EXTRAS_URI")
AZURE_OPENAI_API_KEY: Optional[str] = dotenv.get_key(ENV_PATH, "AZURE_OPENAI_API_KEY")

EXTRA_AZURE_MODELS: dict[str, str] = {
    "jais-30b": "jais-30b-chat",
    "phi-4": "Phi-4-reasoning",
    "llama-4": "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "grok-3": "grok-3",
    "deepseek-r1": "DeepSeek-R1-0528",
    "mai-ds-r1": "MAI-DS-R1",
    }

def call_azure_model(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.8,
    top_p: float = 0.95,
    presence_penalty: float = 0,
    frequency_penalty: float = 0
    ) -> Optional[str]:    
        
    headers: dict[str, str] = { "Content-Type": "application/json", "Authorization": f"Bearer {api_key}" }
    
    data: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "model": EXTRA_AZURE_MODELS[model]
        }
    
    try:
        response: Response = httpx.post(AZURE_EXTRAS_URI, headers=headers, json=data) # type: ignore
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
        
    except RequestError as e:
        print(f"Request error: {e}")
        return None
    except KeyError as e:
        print(f"Response parsing error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Convenience function to call specific models
def prompt_azure_extra(prompt: str, model_key: str) -> Optional[str]:
    """Helper function to call Azure extra models with environment variables"""
    if not AZURE_EXTRAS_URI or not AZURE_OPENAI_API_KEY:
        return "Error: Missing environment variables AZURE_EXTRAS_URI or AZURE_OPENAI_API_KEY"
    
    if model_key not in EXTRA_AZURE_MODELS:
        return f"Error: Model {model_key} not found. Available: {list(EXTRA_AZURE_MODELS.keys())}"
    
    return call_azure_model(prompt=prompt, api_key=AZURE_OPENAI_API_KEY, model=model_key)

# Example usage:
if __name__ == "__main__":
    PROMPT = "What is the name of the largest planet in our solar system?"
    
    # Fixed the list comprehension
    for model_key in EXTRA_AZURE_MODELS.keys():
        print(f"\n=== {model_key} ===")
        result = call_azure_model(
            prompt=PROMPT,
            # uri=AZURE_EXTRAS_URI,  # type: ignore[arg-type]
            api_key=AZURE_OPENAI_API_KEY,  # type: ignore[arg-type]
            model=model_key  # Fixed: was empty string
        )
        print(result or "No response")