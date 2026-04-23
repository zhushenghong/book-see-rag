from langchain_core.language_models import BaseChatModel
from book_see_rag.config import get_settings


def create_llm(provider: str | None = None) -> BaseChatModel:
    settings = get_settings()
    provider = provider or settings.llm_provider

    match provider:
        case "vllm":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.vllm_model,
                openai_api_base=settings.vllm_base_url,
                openai_api_key="EMPTY",
                temperature=0.1,
                max_tokens=2048,
            )
        case "claude":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model_name=settings.claude_model,
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_base_url or None,
                temperature=0.1,
                max_tokens_to_sample=2048,
            )
        case "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.1,
                max_tokens=2048,
            )
        case _:
            raise ValueError(f"未知 LLM provider: {provider}，可选：vllm | claude | openai")
