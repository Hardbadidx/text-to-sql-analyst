"""Application configuration and environment settings."""

import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Hard-coded non-negotiable minimum denylist for sensitive patient columns
MANDATORY_DENYLIST: tuple[str, ...] = ("phone", "email", "address", "ssn")


class Settings(BaseSettings):
    """Central configuration for database, LLM provider, safety policies, and paths."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="sqlite:///clinic.db",
        description="SQLAlchemy database connection URL. Use sqlite:///... or postgresql://...",
    )

    # Security & Safety
    denylist_columns: Annotated[list[str], NoDecode] = Field(
        default=["phone", "email", "address", "ssn", "social_security"],
        description="Column names to exclude completely from LLM context and reject during validation.",
    )
    reference_date: str = Field(
        default="2024-12-31",
        description="Fixed system current date matching dataset boundary for relative date queries.",
    )
    max_query_limit: int = Field(
        default=1000,
        description="Maximum allowed row LIMIT for generated SQL.",
    )
    query_timeout_seconds: float = Field(
        default=5.0,
        description="Maximum execution timeout in seconds before query cancellation.",
    )

    # LLM Settings
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider: 'ollama', 'gemini', or 'mock'.",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        description="Timeout in seconds for LLM provider HTTP requests.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama local API.",
    )
    ollama_model: str = Field(
        default="llama3",
        description="Ollama model identifier (e.g. llama3, qwen2.5-coder).",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key.",
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Google Gemini model identifier.",
    )

    # Logging and Audit
    log_file: Path = Field(
        default=Path("logs/queries.jsonl"),
        description="Path to JSONL audit log file.",
    )

    @field_validator("denylist_columns", mode="before")
    @classmethod
    def parse_and_enforce_denylist(cls, v: object) -> list[str]:
        """Parse comma-separated strings, JSON arrays, or lists, and enforce mandatory denylist."""
        configured: list[str] = []
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                try:
                    parsed = json.loads(v_str)
                    if isinstance(parsed, list):
                        configured = [str(x).strip().lower() for x in parsed if str(x).strip()]
                except Exception:
                    configured = [x.strip().lower() for x in v_str.split(",") if x.strip()]
            elif v_str:
                configured = [x.strip().lower() for x in v_str.split(",") if x.strip()]
        elif isinstance(v, (list, tuple, set)):
            configured = [str(x).strip().lower() for x in v if str(x).strip()]

        # Rule (b): env value adds to hard-coded minimum denylist, never replaces or empties it
        combined = list(MANDATORY_DENYLIST)
        for col in configured:
            if col not in combined:
                combined.append(col)
        return combined


# Singleton instance
settings = Settings()
