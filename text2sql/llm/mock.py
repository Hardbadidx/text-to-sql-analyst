"""Deterministic Mock LLM Provider for unit testing and CI without external dependencies."""

import json
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from text2sql.llm.base import BaseLLMProvider, LLMError

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(BaseLLMProvider):
    """Mock provider allowing sequenced or deterministic scripted responses for tests."""

    def __init__(
        self,
        responses: list[dict[str, Any] | str | Exception] | None = None,
        default_response: dict[str, Any] | str | None = None,
        custom_handler: Callable[[str, str | None], Any] | None = None,
    ):
        self.responses = list(responses or [])
        self.default_response = default_response or {
            "sql": "SELECT COUNT(*) FROM appointments",
            "reasoning_summary": "Counting total appointments.",
            "is_refusal": False,
            "refusal_reason": None,
        }
        self.custom_handler = custom_handler
        self.call_history: list[dict[str, Any]] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.call_history.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.custom_handler:
            res = self.custom_handler(prompt, system_prompt)
            return json.dumps(res) if isinstance(res, dict) else str(res)

        if self.responses:
            next_resp = self.responses.pop(0)
            if isinstance(next_resp, Exception):
                raise next_resp
            return json.dumps(next_resp) if isinstance(next_resp, dict) else str(next_resp)

        return (
            json.dumps(self.default_response)
            if isinstance(self.default_response, dict)
            else str(self.default_response)
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        raw_text = self.generate(prompt, system_prompt)
        try:
            data = json.loads(raw_text)
            return response_model.model_validate(data)
        except Exception as e:
            raise LLMError(f"Failed to parse structured output from mock response: {raw_text}") from e
