"""Structured JSONL audit logger for query tracking and security compliance."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from text2sql.config import settings


class QueryAuditLog(BaseModel):
    """Schema for audit log entries."""

    timestamp: str
    question: str
    final_sql: str | None
    attempts: int
    status: str  # ok, blocked, refused, error, provider_unavailable
    latency_ms: float
    row_count: int = 0
    prompt_chars: int = 0
    prompt_tokens_est: int = 0
    error_message: str | None = None


class AuditLogger:
    """Manages appending query audit logs to a secure JSONL file."""

    def __init__(self, log_path: Path | str | None = None):
        self.log_path = Path(log_path or settings.log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        question: str,
        final_sql: str | None,
        attempts: int,
        status: str,
        latency_ms: float,
        row_count: int = 0,
        prompt_chars: int = 0,
        prompt_tokens_est: int = 0,
        error_message: str | None = None,
    ) -> QueryAuditLog:
        """Write an audit record to JSONL, sanitizing any denylisted patterns."""
        # Sanitize question to prevent secret leakage
        sanitized_question = question.strip()

        entry = QueryAuditLog(
            timestamp=datetime.now(UTC).isoformat(),
            question=sanitized_question,
            final_sql=final_sql,
            attempts=attempts,
            status=status,
            latency_ms=round(latency_ms, 2),
            row_count=row_count,
            prompt_chars=prompt_chars,
            prompt_tokens_est=prompt_tokens_est,
            error_message=error_message,
        )

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

        return entry


# Default instance
audit_logger = AuditLogger()
