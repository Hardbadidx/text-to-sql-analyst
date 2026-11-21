"""Google Gemini provider implementation using the official Google GenAI SDK with retry/backoff."""

import json
import time
from typing import TypeVar

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel

from text2sql.llm.base import BaseLLMProvider, LLMError, LLMRateLimitError

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(BaseLLMProvider):
    """Integrates with Google Gemini via the official Google GenAI SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided for GeminiProvider.")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff_seconds
        self.client = genai.Client(api_key=api_key)

    def _execute_with_backoff(self, fn, *args, **kwargs):
        """Execute callable with exponential backoff on transient rate limits (429)."""
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
                    raise LLMRateLimitError(f"Gemini rate limit exceeded after {self.max_retries} retries: {ae}") from ae
                raise LLMError(f"Gemini API error ({code}): {ae}") from ae
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper():
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise LLMRateLimitError(f"Gemini rate limit exceeded: {e}") from e
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
            parsed = json.loads(raw_text)
            return response_model.model_validate(parsed)

        return self._execute_with_backoff(_call)
