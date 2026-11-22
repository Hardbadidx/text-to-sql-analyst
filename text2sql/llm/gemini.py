"""Google Gemini provider implementation using the official Google GenAI SDK with retry/backoff."""

import json
import re
import time
from typing import TypeVar

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel

from text2sql.llm.base import (
    BaseLLMProvider,
    LLMError,
    ProviderUnavailableError,
    StructuredGenerationError,
)

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(BaseLLMProvider):
    """Integrates with Google Gemini via the official Google GenAI SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-flash-lite",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
        max_total_backoff_seconds: float = 60.0,
    ):
        self.api_key = api_key or ""
        raw_model = (model or "gemini-3.5-flash-lite").strip()
        normalized = re.sub(r"\s+", "-", raw_model).lower()
        if normalized.startswith("models/"):
            normalized = normalized[len("models/"):]
        self.model = normalized
        self.timeout_seconds = timeout_seconds
        self.max_retries = min(max_retries, 3)
        self.initial_backoff = initial_backoff_seconds
        self.max_total_backoff = max_total_backoff_seconds

        http_opts = types.HttpOptions(
            timeout=int(self.timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        self.client = (
            genai.Client(api_key=self.api_key, http_options=http_opts)
            if self.api_key
            else None
        )

    def _execute_with_backoff(self, fn, *args, **kwargs):
        """Execute callable with capped backoff on transient rate limits (429) or spikes (503)."""
        if not self.api_key or self.client is None:
            raise ProviderUnavailableError(
                "No language model is configured on this deployment. Use Demo Mode."
            )

        backoff = self.initial_backoff
        total_backoff = 0.0
        start_time = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except APIError as ae:
                code = getattr(ae, "code", None)
                msg_upper = str(ae).upper()

                # Treat invalid-model / 400 / 404 errors as provider_unavailable (fail fast, no SQL self-correction)
                if code in (400, 404):
                    raise ProviderUnavailableError(
                        f"Gemini model configuration error ({code}): {ae}. Verify GEMINI_MODEL setting."
                    ) from ae

                # Capped retry on rate limits (429) or temporary capacity spikes (503)
                if (
                    code in (429, 503)
                    or "RESOURCE_EXHAUSTED" in msg_upper
                    or "UNAVAILABLE" in msg_upper
                ):
                    # Fail fast with no retries if quota is per-day or RetryInfo delay > 60s
                    is_per_day = any(p in msg_upper for p in ("PERDAY", "PER_DAY", "PER DAY", "DAILY"))
                    delay_match = re.search(
                        r"(?:retryinfo|retry[_\s]?delay|retry[_\s]?after)\D*?(\d+(?:\.\d+)?)\s*s?",
                        str(ae),
                        re.IGNORECASE,
                    )
                    retry_delay = float(delay_match.group(1)) if delay_match else 0.0

                    if is_per_day:
                        raise ProviderUnavailableError(
                            f"Gemini daily API quota reached (quotaId contains PerDay): {ae}"
                        ) from ae

                    if retry_delay > 60.0:
                        raise ProviderUnavailableError(
                            f"Gemini retry delay exceeds 60s ({retry_delay:.0f}s): {ae}"
                        ) from ae

                    total_elapsed = time.perf_counter() - start_time
                    if (
                        attempt < self.max_retries
                        and (total_elapsed + backoff) <= self.max_total_backoff
                    ):
                        time.sleep(backoff)
                        total_backoff += backoff
                        backoff = min(backoff * 2.0, 16.0)
                        continue
                    raise ProviderUnavailableError(
                        f"Gemini rate limit or temporary capacity exceeded (attempts={attempt + 1}, total_backoff={total_backoff:.1f}s): {ae}"
                    ) from ae

                if code in (401, 403, 502, 504):
                    raise ProviderUnavailableError(
                        f"Gemini API authentication/availability error ({code}): {ae}"
                    ) from ae
                raise LLMError(f"Gemini API error ({code}): {ae}") from ae
            except Exception as e:
                err_str = str(e).upper()
                if any(w in err_str for w in ["TIMEOUT", "TIMED OUT", "DEADLINE_EXCEEDED"]):
                    raise ProviderUnavailableError(
                        f"Gemini request timed out after {self.timeout_seconds}s: {e}"
                    ) from e
                if any(w in err_str for w in ["400", "404", "INVALID_ARGUMENT", "NOT_FOUND"]):
                    raise ProviderUnavailableError(
                        f"Gemini model or argument error: {e}. Verify GEMINI_MODEL."
                    ) from e
                if any(x in err_str for x in ["CONNECT", "UNREACHABLE", "NETWORK", "SSL", "API_KEY"]):
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
