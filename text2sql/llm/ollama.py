"""Ollama provider implementation for local, free LLM inference."""

import json
from typing import TypeVar

import httpx
from pydantic import BaseModel

from text2sql.llm.base import BaseLLMProvider, LLMError

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
        except httpx.ConnectError as ce:
            raise LLMError(
                f"Could not connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from ce
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
                parsed_json = json.loads(raw_response)
                return response_model.model_validate(parsed_json)
        except httpx.ConnectError as ce:
            raise LLMError(
                f"Could not connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from ce
        except Exception as e:
            raise LLMError(f"Ollama structured generation failed: {e}") from e
