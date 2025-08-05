import cohere

def rerank(query: str, documents: list[str], top_n: int = 5) -> cohere.V2RerankResponse:
    return cohere.ClientV2().rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_n,
        )

def main() -> None:
    from main import (
        prompt_oss, prompt_kimi, prompt_zai,
        prompt_azure_openai, prompt_cohere
    )
    from extras import prompt_azure_extra

    query = "Who is the first Prime Minister of Jamaica?"
    documents = [
        prompt_oss(query, "20B"),
        prompt_kimi(query),
        prompt_zai(query),
        prompt_azure_openai(query, "gpt-4o"),
        prompt_cohere(query, "command-r7b"),
        prompt_azure_extra(query, "Phi-4-reasoning"),
    ]

    reranked = rerank(query, documents, top_n=3)
    print(reranked)

if __name__ == "__main__":
    main()    
    print("🐬")