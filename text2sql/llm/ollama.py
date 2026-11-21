"""Ollama provider implementation for local, free LLM inference."""

import json
from typing import TypeVar

import httpx
from pydantic import BaseModel

from text2sql.llm.base import (
    BaseLLMProvider,
    LLMError,
    ProviderUnavailableError,
    StructuredGenerationError,
)

T = TypeVar("T", bound=BaseModel)


class OllamaProvider(BaseLLMProvider):
    """Integrates with a local Ollama server instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout_seconds: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout_seconds

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
        except (httpx.ConnectError, httpx.ConnectTimeout) as ce:
            raise ProviderUnavailableError(
                f"Could not connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from ce
        except httpx.TimeoutException as te:
            raise ProviderUnavailableError(
                f"Ollama request timed out after {self.timeout}s: {te}"
            ) from te
        except httpx.HTTPStatusError as se:
            if se.response.status_code in (401, 403, 404, 502, 503, 504):
                raise ProviderUnavailableError(
                    f"Ollama HTTP error ({se.response.status_code}) at {self.base_url}: {se}"
                ) from se
            raise LLMError(f"Ollama HTTP error: {se}") from se
        except Exception as e:
            raise LLMError(f"Ollama generation failed: {e}") from e

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        schema_json = json.dumps(response_model.model_json_schema())
        augmented_system = (
            f"{system_prompt or ''}\n\n"
            f"You MUST respond ONLY with a valid JSON object strictly matching this schema:\n"
            f"{schema_json}\n"
            f"Do not include any other markdown formatting, backticks, or conversational preamble."
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": augmented_system,
            "format": "json",
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_response = data.get("response", "{}")
        except (httpx.ConnectError, httpx.ConnectTimeout) as ce:
            raise ProviderUnavailableError(
                f"Could not connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from ce
        except httpx.TimeoutException as te:
            raise ProviderUnavailableError(
                f"Ollama request timed out after {self.timeout}s: {te}"
            ) from te
        except httpx.HTTPStatusError as se:
            if se.response.status_code in (401, 403, 404, 502, 503, 504):
                raise ProviderUnavailableError(
                    f"Ollama HTTP error ({se.response.status_code}) at {self.base_url}: {se}"
                ) from se
            raise LLMError(f"Ollama HTTP error: {se}") from se
        except Exception as e:
            raise LLMError(f"Ollama request failed: {e}") from e

        try:
            parsed_json = json.loads(raw_response)
            return response_model.model_validate(parsed_json)
        except Exception as e:
            raise StructuredGenerationError(
                f"Failed to parse Ollama response into JSON schema: {e}. Raw response: {raw_response[:200]}"
            ) from e
