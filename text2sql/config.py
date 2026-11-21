"""Application configuration and environment settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    denylist_columns: list[str] = Field(
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
        default="gemini-2.5-flash",
        description="Google Gemini model identifier.",
    )

    # Logging and Audit
    log_file: Path = Field(
        default=Path("logs/queries.jsonl"),
        description="Path to JSONL audit log file.",
    )


# Singleton instance
settings = Settings()
