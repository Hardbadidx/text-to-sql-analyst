"""LLM Provider factory for resolving active provider based on configuration."""

from text2sql.config import Settings, settings
from text2sql.llm.base import BaseLLMProvider
from text2sql.llm.gemini import GeminiProvider
from text2sql.llm.mock import MockLLMProvider
from text2sql.llm.ollama import OllamaProvider


def get_llm_provider(custom_settings: Settings | None = None) -> BaseLLMProvider:
    """Return the configured LLM provider instance."""
    cfg = custom_settings or settings
    provider_name = cfg.llm_provider.lower().strip()

    if provider_name == "gemini":
        if not cfg.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please export GEMINI_API_KEY or configure .env."
            )
        return GeminiProvider(api_key=cfg.gemini_api_key, model=cfg.gemini_model)

    elif provider_name == "ollama":
        return OllamaProvider(base_url=cfg.ollama_base_url, model=cfg.ollama_model)

    elif provider_name == "mock":
        return MockLLMProvider()

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{cfg.llm_provider}'. Supported: 'ollama', 'gemini', 'mock'."
        )
