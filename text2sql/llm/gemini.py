"""Google Gemini provider implementation using the official Google GenAI SDK with retry/backoff."""

import json
import time
from typing import TypeVar

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel

from text2sql.llm.base import (
    BaseLLMProvider,
    LLMError,
    LLMRateLimitError,
    ProviderUnavailableError,
    StructuredGenerationError,
)

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(BaseLLMProvider):
    """Integrates with Google Gemini via the official Google GenAI SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
    ):
        self.api_key = api_key or ""
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff_seconds
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def _execute_with_backoff(self, fn, *args, **kwargs):
        """Execute callable with exponential backoff on transient rate limits (429)."""
        if not self.api_key or self.client is None:
            raise ProviderUnavailableError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your environment or .env file."
            )

        backoff = self.initial_backoff
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except APIError as ae:
                code = getattr(ae, "code", None)
                if code == 429 or "RESOURCE_EXHAUSTED" in str(ae).upper():
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise LLMRateLimitError(
                        f"Gemini rate limit exceeded after {self.max_retries} retries: {ae}"
                    ) from ae
                if code in (401, 403, 404, 502, 503, 504):
                    raise ProviderUnavailableError(
                        f"Gemini API authentication/availability error ({code}): {ae}"
                    ) from ae
                raise LLMError(f"Gemini API error ({code}): {ae}") from ae
            except Exception as e:
                err_str = str(e).upper()
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise LLMRateLimitError(f"Gemini rate limit exceeded: {e}") from e
                if any(x in err_str for x in ["CONNECT", "TIMEOUT", "UNREACHABLE", "NETWORK", "API_KEY"]):
                    raise ProviderUnavailableError(f"Gemini connection/network error: {e}") from e
                raise LLMError(f"Gemini call failed: {e}") from e

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
        )

        def _call():
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return resp.text or ""

        return self._execute_with_backoff(_call)

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=response_model,
            temperature=0.0,
        )

        def _call():
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            raw_text = resp.text or "{}"
            try:
                parsed = json.loads(raw_text)
                return response_model.model_validate(parsed)
            except Exception as e:
                raise StructuredGenerationError(
                    f"Failed to parse Gemini response into JSON schema: {e}. Raw response: {raw_text[:200]}"
                ) from e

        return self._execute_with_backoff(_call)
