"""Unit tests for Streamlit UI pure utility functions and helpers.

Tests message formatting, path/secret sanitization, chart selection,
quota counting, demo response fallback resolution, and eval summary parsing.
Zero live API calls.
"""

import pandas as pd

from app.ui_utils import (
    EXAMPLE_QUESTIONS,
    check_quota_limit,
    format_status_message,
    get_demo_response,
    get_remaining_quota,
    load_eval_summary,
    sanitize_text,
)
from text2sql.charts import auto_pick_chart_type, generate_plotly_figure


def test_example_questions_count_and_structure():
    """EXAMPLE_QUESTIONS must contain exactly 6 curated benchmark questions with required keys."""
    assert len(EXAMPLE_QUESTIONS) == 6
    for q in EXAMPLE_QUESTIONS:
        assert "id" in q
        assert "title" in q
        assert "question" in q
        assert "category" in q
        assert len(q["question"]) > 10


def test_sanitize_text_removes_paths_and_secrets():
    """sanitize_text must strip Windows/Unix paths, stack traces, and API keys."""
    raw = (
        "Error in C:\\Users\\kshit\\Desktop\\Text_To_SQL\\text2sql\\agent.py at line 42.\n"
        "Traceback (most recent call last):\n"
        "  File 'C:\\python\\lib.py', line 10\n"
        "KeyError: 'secret_key'\n"
        "API Key AIzaSyDfakeKey1234567890123456789012 and sk-abcdefghijklmnopqrstuvwxyz1234"
    )
    cleaned = sanitize_text(raw)

    assert "C:\\Users" not in cleaned
    assert "AIzaSyDfakeKey" not in cleaned
    assert "sk-abcdefgh" not in cleaned
    assert "Traceback" not in cleaned
    assert "[path]" in cleaned or "[REDACTED_API_KEY]" in cleaned


def test_format_status_message_refused():
    """Status 'refused' must return a polite warning banner with the sanitized reason."""
    msg = format_status_message(
        status="refused",
        error_message="User requested DROP TABLE",
        refusal_reason="Write operations are forbidden on the read-only clinic database.",
    )
    assert msg["level"] == "warning"
    assert msg["title"] == "Request Refused"
    assert "Write operations are forbidden" in msg["message"]
    assert "safety boundary" in msg["explanation"]


def test_format_status_message_validator_blocked():
    """Validator errors (e.g. denylisted column access) must produce a clean blocked message."""
    msg = format_status_message(
        status="error",
        error_message="SecurityValidationError: Access to denylisted column 'email' is forbidden.",
    )
    assert msg["level"] == "error"
    assert "Security Validator" in msg["title"]
    assert "denylisted column" in msg["message"]


def test_format_status_message_provider_unavailable_quota():
    """Daily quota 429 errors must format into a user-friendly quota message with demo mode suggestion."""
    msg = format_status_message(
        status="provider_unavailable",
        error_message="Resource has been exhausted: Quota exceeded for quotaId: GenerateContentRequestsPerDay",
    )
    assert msg["level"] == "error"
    assert "Quota Reached" in msg["title"]
    assert "Demo Mode" in msg["message"]


def test_format_status_message_ok():
    """Successful queries must produce a success banner."""
    msg = format_status_message(status="ok", error_message=None, explanation="Retrieved 5 rows.")
    assert msg["level"] == "success"
    assert msg["title"] == "Query Executed Successfully"
    assert "Retrieved 5 rows." in msg["explanation"]


def test_check_quota_limit_and_remaining():
    """Quota helpers must accurately calculate quota exhaustion and remaining queries."""
    assert not check_quota_limit(0, max_limit=10)
    assert get_remaining_quota(0, max_limit=10) == 10

    assert not check_quota_limit(5, max_limit=10)
    assert get_remaining_quota(5, max_limit=10) == 5

    assert check_quota_limit(10, max_limit=10)
    assert get_remaining_quota(10, max_limit=10) == 0

    assert check_quota_limit(15, max_limit=10)
    assert get_remaining_quota(15, max_limit=10) == 0


def test_get_demo_response_for_all_example_questions():
    """Every example question in EXAMPLE_QUESTIONS must resolve to a valid demo response."""
    for ex in EXAMPLE_QUESTIONS:
        resp = get_demo_response(ex["question"])
        assert resp is not None, f"Demo response missing for: {ex['question']}"
        assert resp["status"] == "ok"
        assert resp["sql"] is not None and "SELECT" in resp["sql"]
        assert isinstance(resp["df"], pd.DataFrame)
        assert len(resp["df"]) > 0
        assert resp["chart_type"] in ("bar", "line", "pie", "table-only")


def test_get_demo_response_unknown_question():
    """Random unknown question not in cache should return None."""
    resp = get_demo_response("What is the temperature in Antarctica right now at this second?")
    assert resp is None


def test_load_eval_summary_from_latest_json():
    """load_eval_summary must parse metrics from eval/results/latest.json."""
    summary = load_eval_summary("eval/results/latest.json")
    assert summary["model"] != "Unknown"
    assert summary["total_questions"] == 56
    assert summary["exact_accuracy_pct"] >= 70.0
    assert summary["lenient_accuracy_pct"] >= 90.0
    assert summary["safety_block_rate_pct"] == 100.0
    assert "Refused correctly" in summary["policy_refusal_status"]


def test_load_eval_summary_missing_file_fallback(tmp_path):
    """load_eval_summary must return graceful defaults if latest.json does not exist."""
    missing = tmp_path / "non_existent_results.json"
    summary = load_eval_summary(missing)
    assert isinstance(summary, dict)
    assert "exact_accuracy_pct" in summary
    assert summary["total_questions"] == 56


def test_chart_selection_and_figure_generation_integration():
    """Ensure auto-pick and Plotly figure generation integrate cleanly for the UI."""
    df_bar = pd.DataFrame({"doctor": [f"Dr. {i}" for i in range(10)], "revenue": [5000.0 + i for i in range(10)]})
    assert auto_pick_chart_type(df_bar) == "bar"
    fig_bar = generate_plotly_figure(df_bar, chart_type="bar")
    assert fig_bar is not None

    df_pie = pd.DataFrame({"payment_method": ["credit_card", "cash", "insurance"], "amount": [2000.0, 500.0, 1500.0]})
    assert auto_pick_chart_type(df_pie) == "pie"
    fig_pie = generate_plotly_figure(df_pie, chart_type="pie")
    assert fig_pie is not None

    df_line = pd.DataFrame({"month": ["2024-01", "2024-02"], "invoiced": [1000.0, 1200.0]})
    assert auto_pick_chart_type(df_line) == "line"
    fig_line = generate_plotly_figure(df_line, chart_type="line")
    assert fig_line is not None

    df_single = pd.DataFrame({"total": [42]})
    assert auto_pick_chart_type(df_single) == "table-only"
    assert generate_plotly_figure(df_single) is None
