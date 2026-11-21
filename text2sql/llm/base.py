"""Abstract base class interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM generation failures."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when an LLM provider encounters a rate limit or quota ceiling."""

    pass


class BaseLLMProvider(ABC):
    """Universal interface for local and API-based LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate unstructured text completion."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generate structured output adhering strictly to a Pydantic schema."""
        pass
