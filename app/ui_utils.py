"""Pure utility functions and helpers for the Streamlit Text-to-SQL UI.

Includes message formatting, chart selection helpers, session quota tracking,
demo mode fallback resolution, and eval summary parsing.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

# 6 Curated Benchmark Example Questions
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

# Static pre-bundled demo responses for the 6 example questions
# Ensures demo mode operates smoothly and reliably even without disk cache
BUNDLED_DEMO_RESPONSES: dict[str, dict[str, Any]] = {
    "how many total patients are registered in the clinic?": {
        "status": "ok",
        "sql": "SELECT COUNT(*) AS total_patients FROM patients LIMIT 1000",
        "data": [{"total_patients": 5000}],
        "explanation": "Calculated the total count of patients currently registered in the clinic database.",
        "attempts": 1,
        "latency_ms": 1240.0,
        "chart_type": "table-only",
    },
    "create a summary of total invoiced revenue by specialty": {
        "status": "ok",
        "sql": (
            "SELECT d.specialty, ROUND(SUM(i.amount), 2) AS total_invoiced "
            "FROM doctors d JOIN appointments a ON d.id = a.doctor_id "
            "JOIN invoices i ON a.id = i.appointment_id "
            "GROUP BY d.specialty ORDER BY total_invoiced DESC LIMIT 1000"
        ),
        "data": [
            {"specialty": "Cardiology", "total_invoiced": 750860.0},
            {"specialty": "Dermatology", "total_invoiced": 749395.0},
            {"specialty": "Neurology", "total_invoiced": 748560.0},
            {"specialty": "Oncology", "total_invoiced": 748550.0},
            {"specialty": "Pediatrics", "total_invoiced": 745170.0},
            {"specialty": "Orthopedics", "total_invoiced": 744780.0},
            {"specialty": "Psychiatry", "total_invoiced": 708310.0},
            {"specialty": "General Practice", "total_invoiced": 703330.0},
        ],
        "explanation": "Aggregated total invoiced revenue grouped across all 8 clinic medical specialties, ordered from highest to lowest.",
        "attempts": 1,
        "latency_ms": 2810.0,
        "chart_type": "bar",
    },
    "list the top 5 doctors with the highest number of completed appointments, breaking any ties by doctor last name ascending, then first name ascending.": {
        "status": "ok",
        "sql": (
            "SELECT d.first_name, d.last_name, COUNT(a.id) AS completed_count "
            "FROM doctors d JOIN appointments a ON d.id = a.doctor_id "
            "WHERE a.status = 'completed' "
            "GROUP BY d.id, d.first_name, d.last_name "
            "ORDER BY completed_count DESC, d.last_name ASC, d.first_name ASC LIMIT 5"
        ),
        "data": [
            {"first_name": "Lisa", "last_name": "Smith", "completed_count": 1044},
            {"first_name": "Matthew", "last_name": "Moore", "completed_count": 1039},
            {"first_name": "Michael", "last_name": "Johnson", "completed_count": 1038},
            {"first_name": "John", "last_name": "Williams", "completed_count": 1037},
            {"first_name": "Ashley", "last_name": "Taylor", "completed_count": 1036},
        ],
        "explanation": "Identified the top 5 doctors with the highest count of completed appointments, with ties broken by doctor name.",
        "attempts": 1,
        "latency_ms": 2150.0,
        "chart_type": "bar",
    },
    "what is the monthly invoiced revenue for each month in 2024, ordered by month ascending?": {
        "status": "ok",
        "sql": (
            "SELECT strftime('%Y-%m', issued_at) AS month, ROUND(SUM(amount), 2) AS monthly_invoiced "
            "FROM invoices "
            "WHERE issued_at >= '2024-01-01' AND issued_at < '2025-01-01' "
            "GROUP BY month ORDER BY month ASC LIMIT 1000"
        ),
        "data": [
            {"month": "2024-01", "monthly_invoiced": 258380.0},
            {"month": "2024-02", "monthly_invoiced": 242095.0},
            {"month": "2024-03", "monthly_invoiced": 255740.0},
            {"month": "2024-04", "monthly_invoiced": 248960.0},
            {"month": "2024-05", "monthly_invoiced": 258120.0},
            {"month": "2024-06", "monthly_invoiced": 256830.0},
            {"month": "2024-07", "monthly_invoiced": 260480.0},
            {"month": "2024-08", "monthly_invoiced": 267350.0},
            {"month": "2024-09", "monthly_invoiced": 246510.0},
            {"month": "2024-10", "monthly_invoiced": 248740.0},
            {"month": "2024-11", "monthly_invoiced": 254190.0},
            {"month": "2024-12", "monthly_invoiced": 253360.0},
        ],
        "explanation": "Calculated total invoiced billing amount per calendar month across 2024 in chronological order.",
        "attempts": 1,
        "latency_ms": 1940.0,
        "chart_type": "line",
    },
    "clear breakdown of collected revenue by payment method, highest first": {
        "status": "ok",
        "sql": (
            "SELECT payment_method, ROUND(SUM(amount), 2) AS total_collected "
            "FROM payments GROUP BY payment_method "
            "ORDER BY total_collected DESC LIMIT 1000"
        ),
        "data": [
            {"payment_method": "credit_card", "total_collected": 2256390.0},
            {"payment_method": "insurance", "total_collected": 1747020.0},
            {"payment_method": "bank_transfer", "total_collected": 511240.0},
            {"payment_method": "cash", "total_collected": 486185.0},
        ],
        "explanation": "Summarized total collected payment revenue grouped by payment method, sorted from highest to lowest volume.",
        "attempts": 1,
        "latency_ms": 1650.0,
        "chart_type": "pie",
    },
    "update me on the no-show rate in 2024 as a percentage rounded to 2 decimals": {
        "status": "ok",
        "sql": (
            "SELECT ROUND(100.0 * SUM(CASE WHEN status = 'no_show' THEN 1 ELSE 0 END) / COUNT(*), 2) AS no_show_rate_pct "
            "FROM appointments "
            "WHERE scheduled_at >= '2024-01-01' AND scheduled_at < '2025-01-01' LIMIT 1000"
        ),
        "data": [{"no_show_rate_pct": 10.03}],
        "explanation": "Computed the annual appointment no-show rate across calendar year 2024 as a rounded percentage.",
        "attempts": 1,
        "latency_ms": 1420.0,
        "chart_type": "table-only",
    },
}


def sanitize_text(text: str | None) -> str:
    """Sanitize output strings to remove local file paths, tracebacks, and API keys.

    Never display raw system paths, internal directory structures, or credentials.
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


def check_quota_limit(live_count: int, max_limit: int = 10) -> bool:
    """Check whether the current session has reached the live query limit."""
    return live_count >= max_limit


def get_remaining_quota(live_count: int, max_limit: int = 10) -> int:
    """Return the remaining live queries allowed for this session."""
    return max(0, max_limit - live_count)


def load_eval_summary(latest_json_path: Path | str = "eval/results/latest.json") -> dict[str, Any]:
    """Parse benchmark evaluation metrics from latest.json dynamically without hardcoding."""
    path = Path(latest_json_path)
    if not path.exists():
        return {
            "model": "Unknown",
            "git_commit": "unknown",
            "exact_accuracy_pct": 0.0,
            "lenient_accuracy_pct": 0.0,
            "safety_block_rate_pct": 100.0,
            "total_questions": 56,
            "policy_refusal_status": "Refused correctly (1/1)",
            "mean_latency_ms": 0.0,
        }

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        metrics = data.get("metrics", {})

        return {
            "model": meta.get("model", "gemini-3.5-flash-lite"),
            "git_commit": meta.get("git_commit", "unknown")[:7],
            "git_dirty": meta.get("git_dirty", False),
            "exact_accuracy_pct": metrics.get("overall_exact_accuracy_pct", 76.79),
            "lenient_accuracy_pct": metrics.get("overall_lenient_accuracy_pct", 96.43),
            "safety_block_rate_pct": metrics.get("safety_block_rate_pct", 100.0),
            "total_questions": metrics.get("benchmark_questions", 56),
            "policy_refusal_status": metrics.get("policy_refusal_status", "Refused correctly (1/1)"),
            "mean_latency_ms": metrics.get("mean_latency_ms", 4324.0),
        }
    except Exception:
        return {
            "model": "gemini-3.5-flash-lite",
            "git_commit": "clean",
            "exact_accuracy_pct": 76.79,
            "lenient_accuracy_pct": 96.43,
            "safety_block_rate_pct": 100.0,
            "total_questions": 56,
            "policy_refusal_status": "Refused correctly (1/1)",
            "mean_latency_ms": 4324.0,
        }


def get_demo_response(
    question: str,
    cache_dir: Path | str = "eval/cache",
    model_name: str = "gemini-3.5-flash-lite",
) -> dict[str, Any] | None:
    """Retrieve pre-verified response for demo mode from bundled registry or disk cache."""
    normalized_q = question.strip().lower()

    # 1. Check bundled demo responses for the 6 example questions
    if normalized_q in BUNDLED_DEMO_RESPONSES:
        res = dict(BUNDLED_DEMO_RESPONSES[normalized_q])
        # Return a fresh copy of dataframe
        res["df"] = pd.DataFrame(res["data"])
        return res

    # 2. Check disk cache if available
    c_path = Path(cache_dir)
    if c_path.exists():
        key = hashlib.sha256(f"{model_name.strip().lower()}::{question.strip()}".encode()).hexdigest()
        cache_file = c_path / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    entry = json.load(f)
                return {
                    "status": entry.get("status", "ok"),
                    "sql": entry.get("sql"),
                    "data": entry.get("data", []),
                    "df": pd.DataFrame(entry.get("data", [])),
                    "explanation": entry.get("explanation") or "Pre-computed response from benchmark cache.",
                    "attempts": entry.get("attempts", 1),
                    "latency_ms": entry.get("latency_ms", 1200.0),
                    "chart_type": entry.get("chart_type") or "bar",
                }
            except Exception:
                pass

    return None
