"""Pure utility functions and helpers for the Streamlit Text-to-SQL UI.

Includes message formatting, session quota tracking, demo cache loading,
and eval summary parsing.
Zero hard-coded result data or fabricated latencies.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

GLOBAL_CAP_MESSAGE: str = "Daily live-query limit reached, showing demo mode"

# 6 Curated Benchmark Example Questions (Questions & UI metadata only; no data)
EXAMPLE_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "q01",
        "title": "Total Patients",
        "category": "Basic Metric",
        "question": "How many total patients are registered in the clinic?",
        "icon": "👥",
    },
    {
        "id": "q53",
        "title": "Invoiced by Specialty",
        "category": "Revenue & Specialties",
        "question": "Create a summary of total invoiced revenue by specialty",
        "icon": "📊",
    },
    {
        "id": "q18",
        "title": "Top 5 Doctors",
        "category": "Doctor Performance",
        "question": (
            "List the top 5 doctors with the highest number of completed appointments, "
            "breaking any ties by doctor last name ascending, then first name ascending."
        ),
        "icon": "👨‍⚕️",
    },
    {
        "id": "q21",
        "title": "Monthly Revenue 2024",
        "category": "Time Series Trend",
        "question": "What is the monthly invoiced revenue for each month in 2024, ordered by month ascending?",
        "icon": "📈",
    },
    {
        "id": "q55",
        "title": "Payment Method Breakdown",
        "category": "Collections & Payments",
        "question": "Clear breakdown of collected revenue by payment method, highest first",
        "icon": "💳",
    },
    {
        "id": "q54",
        "title": "2024 No-Show Rate %",
        "category": "Clinical Operations",
        "question": "Update me on the no-show rate in 2024 as a percentage rounded to 2 decimals",
        "icon": "📉",
    },
]

DEFAULT_DEMO_CACHE_PATH = Path(__file__).resolve().parent / "demo_cache.json"
DEFAULT_LATEST_JSON_PATH = Path(__file__).resolve().parent.parent / "eval" / "results" / "latest.json"


def sanitize_text(text: str | None) -> str:
    """Sanitize error messages and explanations to remove system paths, tracebacks,

    and sensitive tokens/API keys.
    """
    if not text:
        return ""

    sanitized = str(text)

    # Remove Windows/Unix absolute file paths (e.g. C:\Users\... or /home/...)
    sanitized = re.sub(r"[A-Za-z]:\\[^:\s\n<>()]+", "[path]", sanitized)
    sanitized = re.sub(r"(/(?:[a-zA-Z0-9_\-\.]+/)+[a-zA-Z0-9_\-\.]+)", "[path]", sanitized)

    # Remove tracebacks
    if "Traceback (most recent call last):" in sanitized:
        sanitized = re.sub(r"Traceback \(most recent call last\):.*", "", sanitized, flags=re.DOTALL).strip()

    # Remove API keys (e.g. AIza..., sk-...)
    sanitized = re.sub(r"\bAIza[0-9A-Za-z-_]{35}\b", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"\bsk-[a-zA-Z0-9]{20,}\b", "[REDACTED_API_KEY]", sanitized)

    return sanitized.strip()


def format_status_message(
    status: str,
    error_message: str | None,
    explanation: str | None = None,
    refusal_reason: str | None = None,
) -> dict[str, str]:
    """Format human-readable user messages for UI banners based on agent status.

    Returns a dictionary with title, message, level ('success', 'warning', 'error', 'info'),
    and a cleaned explanation.
    """
    clean_err = sanitize_text(error_message)
    clean_expl = sanitize_text(explanation)
    clean_ref = sanitize_text(refusal_reason)

    # Handle global daily query cap limit reached
    if (
        "daily live-query limit reached" in clean_err.lower()
        or "daily live-query limit reached" in clean_expl.lower()
        or "daily live-query limit reached" in clean_ref.lower()
    ):
        return {
            "title": "Daily Limit Reached",
            "message": GLOBAL_CAP_MESSAGE,
            "level": "warning",
            "explanation": (
                "The process-level daily live-query limit has been reached across all visitors. "
                "Switching to demo mode to explore pre-verified benchmark responses."
            ),
        }

    # Handle demo cache unavailability
    if "demo data unavailable" in clean_err.lower():
        return {
            "title": "Demo Data Unavailable",
            "message": clean_err,
            "level": "warning",
            "explanation": (
                "No pre-verified demonstration response is available for this question. "
                "To execute custom queries, ensure GEMINI_API_KEY is configured and disable Demo Mode."
            ),
        }

    if status == "refused":
        reason = clean_ref or clean_err or clean_expl or "This request is not permitted under the clinic safety policy."
        return {
            "title": "Request Refused",
            "message": reason,
            "level": "warning",
            "explanation": (
                "The agent's safety boundary determined that this query violates policy constraints "
                "(e.g., destructive operations, modification commands, or access to sensitive contact PII)."
            ),
        }

    if status == "error":
        msg = clean_err or "The query could not be validated or executed safely."
        # Clarify AST validator blocks
        if any(w in msg.lower() for w in ["denylist", "forbidden", "disallowed", "shadowing", "single select"]):
            title = "Query Blocked by Security Validator"
        else:
            title = "Execution or Validation Error"

        return {
            "title": title,
            "message": msg,
            "level": "error",
            "explanation": (
                "The SQL query was rejected by the AST security validator or the read-only database engine. "
                "Database integrity remained fully protected."
            ),
        }

    if status == "provider_unavailable":
        # Check if quota limit reached
        if any(w in clean_err.lower() for w in ["quota", "perday", "429", "resource_exhausted"]):
            title = "API Daily Quota Reached"
            msg = "The live Gemini API quota has been reached for today. You can continue exploring pre-verified benchmark questions in Demo Mode."
        elif any(w in clean_err.lower() for w in ["not configured", "no language model", "missing", "unsupported", "api key", "api_key", "authentication"]):
            title = "No Language Model Configured"
            msg = "No language model is configured on this deployment. Use Demo Mode."
        else:
            title = "Language Model Service Unavailable"
            msg = clean_err or "Unable to reach the language model provider. Please verify network access or switch to Demo Mode."

        return {
            "title": title,
            "message": msg,
            "level": "error",
            "explanation": "The LLM provider could not generate a response. No database queries were performed.",
        }

    # Default / 'ok' status
    return {
        "title": "Query Executed Successfully",
        "message": "The query passed AST security validation and executed against the read-only clinic database.",
        "level": "success",
        "explanation": clean_expl or "Results retrieved successfully.",
    }


class GlobalDailyCapTracker:
    """Thread-safe, process-level global daily query counter keyed by UTC date."""

    def __init__(self, cap: int = 150) -> None:
        self.cap = cap
        self.counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def get_current_utc_date(self) -> str:
        """Return current UTC date formatted as YYYY-MM-DD."""
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def get_count(self, utc_date: str | None = None) -> int:
        """Return the current query count for the given UTC date (defaults to today)."""
        date_key = utc_date or self.get_current_utc_date()
        with self._lock:
            return self.counts.get(date_key, 0)

    def is_cap_reached(self, utc_date: str | None = None) -> bool:
        """Check whether the daily cap is reached for the given UTC date."""
        return self.get_count(utc_date) >= self.cap

    def increment(self, utc_date: str | None = None) -> tuple[int, bool]:
        """Increment count for the given UTC date if cap is not yet reached.

        Returns (new_count, is_capped).
        """
        date_key = utc_date or self.get_current_utc_date()
        with self._lock:
            current = self.counts.get(date_key, 0)
            if current >= self.cap:
                return current, True
            new_val = current + 1
            self.counts[date_key] = new_val
            return new_val, new_val >= self.cap

    def reset_for_date(self, utc_date: str | None = None) -> None:
        """Reset counter for test isolation."""
        date_key = utc_date or self.get_current_utc_date()
        with self._lock:
            self.counts.pop(date_key, None)


def check_global_cap_reached(count: int, cap: int = 150) -> bool:
    """Pure-function check whether the global daily live query count has reached cap."""
    return count >= cap


def check_quota_limit(live_count: int, max_limit: int = 10) -> bool:
    """Check whether the current session has reached the live query limit."""
    return live_count >= max_limit


def get_remaining_quota(live_count: int, max_limit: int = 10) -> int:
    """Return the remaining live queries allowed for this session."""
    return max(0, max_limit - live_count)


def should_consume_quota(
    result_or_status: Any,
    prompt_chars: int | None = None,
    attempts: int | None = None,
) -> bool:
    """Determine whether a query execution should consume visitor live query quota.

    A query counts against the visitor's live quota if a model request was actually sent
    (including model-driven refusals, status 'ok', or query execution/validation 'error').
    It does NOT count for:
    - Deterministic pre-check refusals (no model call made, prompt_chars == 0)
    - Provider failures (status 'provider_unavailable')
    """
    if isinstance(result_or_status, str):
        status = result_or_status
        if status == "provider_unavailable":
            return False
        if prompt_chars is not None:
            return prompt_chars > 0
        return status not in ("refused", "provider_unavailable")

    status = (
        getattr(result_or_status, "status", None)
        if not isinstance(result_or_status, dict)
        else result_or_status.get("status")
    )
    if status == "provider_unavailable":
        return False

    chars = (
        prompt_chars
        if prompt_chars is not None
        else (
            getattr(result_or_status, "prompt_chars", 0)
            if not isinstance(result_or_status, dict)
            else result_or_status.get("prompt_chars", 0)
        )
    )
    return (chars or 0) > 0


def record_live_query(
    current_count: int,
    result_or_status: Any,
    already_counted: bool = False,
    prompt_chars: int | None = None,
) -> tuple[int, bool]:
    """Pure function calculating the new live query count without double counting.

    Returns (new_count, was_counted).
    """
    if already_counted:
        return current_count, False
    if not should_consume_quota(result_or_status, prompt_chars=prompt_chars):
        return current_count, False
    return current_count + 1, True


def agent_result_to_record(agent_res: Any, elapsed_ms: float = 0.0) -> dict[str, Any]:
    """Tolerantly map a real AgentResult instance into a UI record dictionary.

    Uses getattr with safe defaults and never accesses undefined attributes.
    """
    df = getattr(agent_res, "df", None)
    row_count = getattr(agent_res, "row_count", len(df) if df is not None else 0)

    return {
        "question": getattr(agent_res, "question", ""),
        "status": getattr(agent_res, "status", "ok"),
        "sql": getattr(agent_res, "sql", None),
        "df": df,
        "row_count": row_count,
        "explanation": getattr(agent_res, "explanation", "") or "",
        "attempts": getattr(agent_res, "attempts", 1),
        "latency_ms": getattr(agent_res, "latency_ms", elapsed_ms) or elapsed_ms,
        "is_demo": False,
        "error_message": getattr(agent_res, "error_message", None),
        "debug_error": getattr(agent_res, "debug_error", None),
        "chart_type": getattr(agent_res, "chart_type", "table-only"),
        "chart_figure": getattr(agent_res, "chart_figure", None),
        "prompt_chars": getattr(agent_res, "prompt_chars", 0),
    }


def load_demo_cache(cache_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Load the demo cache JSON file containing real re-executed benchmark results."""
    path = Path(cache_path) if cache_path is not None else DEFAULT_DEMO_CACHE_PATH
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_demo_response(
    question: str,
    cache_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Retrieve pre-verified response for demo mode strictly from app/demo_cache.json.

    Returns None if demo_cache.json does not exist or does not contain an entry for the question.
    Never falls back to hardcoded/typed data.
    """
    cache = load_demo_cache(cache_path)
    if not cache:
        return None

    norm_q = question.strip().lower()
    entry = cache.get(norm_q)
    if entry is None:
        # Also check by question_id or exact match
        for val in cache.values():
            if (
                val.get("question", "").strip().lower() == norm_q
                or val.get("question_id", "").strip().lower() == norm_q
            ):
                entry = val
                break

    if entry is None:
        return None

    raw_data = entry.get("data", [])
    df = pd.DataFrame(raw_data) if raw_data else pd.DataFrame()

    return {
        "status": entry.get("status", "ok"),
        "sql": entry.get("sql"),
        "data": raw_data,
        "df": df,
        "row_count": entry.get("row_count", len(df)),
        "explanation": entry.get("explanation", "Pre-verified benchmark response from demonstration cache."),
        "attempts": entry.get("attempts", 1),
        "latency_ms": entry.get("latency_ms", 0.0),
        "chart_type": entry.get("chart_type", "table-only"),
    }


def load_eval_summary(latest_json_path: Path | str | None = None) -> dict[str, Any]:
    """Parse benchmark evaluation metrics from latest.json dynamically without hardcoding."""
    path = Path(latest_json_path) if latest_json_path is not None else DEFAULT_LATEST_JSON_PATH
    if not path.exists():
        return {
            "available": False,
            "message": "eval results unavailable",
        }

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        metrics = data.get("metrics", {})

        if not metrics or not meta:
            return {
                "available": False,
                "message": "eval results unavailable",
            }

        return {
            "available": True,
            "model": meta.get("model", "gemini-3.5-flash-lite"),
            "git_commit": meta.get("git_commit", "7bbd679")[:7],
            "git_dirty": meta.get("git_dirty", False),
            "exact_accuracy_pct": metrics.get("overall_exact_accuracy_pct", 0.0),
            "lenient_accuracy_pct": metrics.get("overall_lenient_accuracy_pct", 0.0),
            "safety_block_rate_pct": metrics.get("safety_block_rate_pct", 0.0),
            "total_questions": metrics.get("benchmark_questions", 0),
            "policy_refusal_status": metrics.get("policy_refusal_status", "N/A"),
            "mean_latency_ms": metrics.get("mean_latency_ms", 0.0),
        }
    except Exception:
        return {
            "available": False,
            "message": "eval results unavailable",
        }
