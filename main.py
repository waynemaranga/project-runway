from openai import OpenAI, AzureOpenAI
import cohere
import dotenv
from pathlib import Path
from typing import Optional

from openai.types.chat.chat_completion import ChatCompletion

HUGGINGFACE_TOKEN: Optional[str] = dotenv.get_key(str(Path(__file__).resolve().parent / ".env"), "HUGGINGFACE_TOKEN")
AZURE_OPENAI_API_KEY: Optional[str] = dotenv.get_key(str(Path(__file__).resolve().parent / ".env"), "AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT: Optional[str] = dotenv.get_key(str(Path(__file__).resolve().parent / ".env"), "AZURE_OPENAI_ENDPOINT")
COHERE_API_KEY: Optional[str] = dotenv.get_key(str(Path(__file__).resolve().parent / ".env"), "COHERE_API_KEY")

oss = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HUGGINGFACE_TOKEN)
o4_mini = AzureOpenAI(api_version="2025-04-01-preview", azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, azure_deployment="o4-mini") # type: ignore[call-arg]
gpt_4_1 = AzureOpenAI(api_version="2025-04-01-preview", azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, azure_deployment="gpt-4.1") # type: ignore[call-arg]
o1 = AzureOpenAI(api_version="2025-04-01-preview", azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, azure_deployment="o1") # type: ignore[call-arg]
gpt_4o = AzureOpenAI(api_version="2025-04-01-preview", azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, azure_deployment="gpt-4o") # type: ignore[call-arg]
co = cohere.ClientV2(api_key=COHERE_API_KEY)

HUGGINGFACE_MODELS: dict[str, str] = {
    "oss-120b": "openai/gpt-oss-120b", # https://openai.com/index/introducing-gpt-oss/
    "oss-20b": "openai/gpt-oss-20b", # https://openai.com/index/introducing-gpt-oss/
    "kimi-k2": "moonshotai/Kimi-K2-Instruct", # https://moonshotai.github.io/Kimi-K2/
    "zai-glm-4.5": "zai-org/GLM-4.5", # https://z.ai/blog/glm-4.5
    } 

AZURE_OPENAI_CLIENTS: dict[str, AzureOpenAI] = {
    "o4-mini": o4_mini,
    "gpt-4.1": gpt_4_1,
    "o1": o1,
    "gpt-4o": gpt_4o,
    }

COHERE_MODELS: dict[str, str] = {
    "command-r7b": "command-r7b-12-2024", 
    "command-r+": "command-r-plus-08-2024",
    "command-a": "command-a-03-2025",
    }

def prompt_oss(prompt: str, model: str = "120B" or "20B") -> Optional[str]:
    response: ChatCompletion = oss.chat.completions.create(
        model="openai/gpt-oss-120b" if model == "120B" else "openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def prompt_kimi(prompt: str)-> Optional[str]:
    response: ChatCompletion = oss.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def prompt_zai(prompt: str)-> Optional[str]:
    response: ChatCompletion = oss.chat.completions.create(
        model="zai-org/GLM-4.5",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def prompt_azure_openai(prompt: str, model: str) -> Optional[str]:
    if model not in AZURE_OPENAI_CLIENTS:
        raise ValueError(f"Model {model} is not supported. Choose from {list(AZURE_OPENAI_CLIENTS.keys())}.")
    
    client: AzureOpenAI = AZURE_OPENAI_CLIENTS[model]
    response: ChatCompletion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def prompt_cohere(prompt: str, model: str):
    response: cohere.V2ChatResponse = co.chat(
        model=COHERE_MODELS[model],
        messages=[{ "role": "user", "content": prompt }] # type: ignore[call-arg]
        )
    
    return response.message.content

def main() -> None:
    print("Hello from project-runway!")
    # prompt = "Determine all possible values of the expression A³ + B³ + C³ — 3ABC where A, B, and C are nonnegative integers."
    prompt = "A red house is made of red bricks, a blue house is made of blue bricks. What is a greenhouse made of? Explain your reasoning."
    
    # Example usage
    # print("oss-120b:\n", prompt_oss(prompt, "120B"), "\n", 60*"-")
    # print("oss-20b:\n", prompt_oss(prompt, "20B"), "\n",60*"-")
    # print("kimi-k2:\n", prompt_kimi(prompt), "\n",60*"-")
    # print("zai-glm-4.5:\n", prompt_zai(prompt), "\n",60*"-")
    # print("o4-mini:\n", prompt_azure_openai(prompt, "o4-mini"), "\n",60*"-")
    # print("gpt-4.1:\n", prompt_azure_openai(prompt, "gpt-4.1"), "\n",60*"-")
    # print("o1:\n", prompt_azure_openai(prompt, "o1"), "\n",60*"-")
    # print("gpt-4o:\n", prompt_azure_openai(prompt, "gpt-4o"), "\n",60*"-")
    print("command-r7b:\n", prompt_cohere(prompt, "command-r7b"), "\n",60*"-")
    print("command-r+:\n", prompt_cohere(prompt, "command-r+"), "\n",60*"-")
    print("command-a:\n", prompt_cohere(prompt, "command-a"), "\n",60*"-")



if __name__ == "__main__":
    main()
    print("🐬")
