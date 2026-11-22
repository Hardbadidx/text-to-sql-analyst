"""Unit tests for Streamlit UI utility functions, AgentResult mapping, and AppTest smoke tests.

Tests:
- Real AgentResult instances for each status (success, refused, blocked, provider_unavailable).
- Mapping real AgentResult instances into UI records via agent_result_to_record.
- Strict detection failing if the UI reads any attribute not defined on AgentResult.
- Human-readable banner message formatting with path/secret sanitization.
- Quota counting and session limit management.
- Pre-bundled demo response fallback for all 6 benchmark examples.
- Dynamic parsing of eval/results/latest.json (verifying commit 7bbd679).
- Streamlit AppTest smoke test loading streamlit_app.py in demo mode and clicking ex_q01.
Zero live API calls.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from app.ui_utils import (
    EXAMPLE_QUESTIONS,
    agent_result_to_record,
    check_quota_limit,
    format_status_message,
    get_demo_response,
    get_remaining_quota,
    load_eval_summary,
    sanitize_text,
)
from text2sql.agent import AgentResult
from text2sql.charts import auto_pick_chart_type, generate_plotly_figure

# ---------------------------------------------------------------------------
# Real AgentResult Test Fixtures (Strict instances of AgentResult model)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_ok_result() -> AgentResult:
    """Real AgentResult for a successful query."""
    return AgentResult(
        question="How many total patients are registered in the clinic?",
        sql="SELECT COUNT(*) AS total_patients FROM patients LIMIT 1000",
        df=pd.DataFrame([{"total_patients": 5000}]),
        row_count=1,
        explanation="Counted all registered clinic patients.",
        chart_type="table-only",
        attempts=1,
        status="ok",
        latency_ms=1240.0,
        prompt_chars=120,
        prompt_tokens_est=30,
    )


@pytest.fixture
def real_refused_result() -> AgentResult:
    """Real AgentResult for a refused query (e.g. write/destructive request)."""
    msg = "Write or destructive operations are not permitted on the read-only clinic database."
    return AgentResult(
        question="Drop table appointments",
        sql=None,
        df=None,
        row_count=0,
        explanation=msg,
        chart_type="table-only",
        attempts=1,
        status="refused",
        latency_ms=3.5,
        prompt_chars=25,
        prompt_tokens_est=6,
        error_message=msg,
    )


@pytest.fixture
def real_blocked_result() -> AgentResult:
    """Real AgentResult for a query blocked by the AST security validator."""
    err = "SecurityValidationError: Access to denylisted column 'email' is forbidden."
    return AgentResult(
        question="List all patient emails",
        sql=None,
        df=None,
        row_count=0,
        explanation="",
        chart_type="table-only",
        attempts=1,
        status="error",
        latency_ms=1520.0,
        prompt_chars=180,
        prompt_tokens_est=45,
        error_message=err,
    )


@pytest.fixture
def real_provider_unavailable_result() -> AgentResult:
    """Real AgentResult when LLM provider encounters quota exhaustion (429)."""
    human_err = "429 Resource has been exhausted (quota exceeded)"
    raw_debug = "ResourceExhausted: 429 Quota exceeded for quotaId: GenerateContentRequestsPerDay"
    return AgentResult(
        question="What is total invoiced revenue in 2024?",
        sql=None,
        df=None,
        row_count=0,
        explanation=f"LLM provider is currently unavailable: {human_err}",
        chart_type="table-only",
        attempts=1,
        status="provider_unavailable",
        latency_ms=42.0,
        prompt_chars=220,
        prompt_tokens_est=55,
        error_message=human_err,
        debug_error=raw_debug,
    )


# ---------------------------------------------------------------------------
# Strict Attribute Verification Helpers
# ---------------------------------------------------------------------------


class DisallowedAttributeAccessError(Exception):
    """Raised whenever code attempts to access an attribute not defined on AgentResult."""


class StrictAgentResultProxy:
    """Strict proxy wrapping a real AgentResult.

    Any attempt to read an attribute not defined in AgentResult.model_fields raises
    DisallowedAttributeAccessError. Because it overrides __getattribute__, it cannot
    be bypassed or suppressed by standard getattr(obj, attr, default).
    """

    def __init__(self, target: AgentResult):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_allowed", set(AgentResult.model_fields.keys()))

    def __getattribute__(self, name: str):
        if name in ("_target", "_allowed", "__class__"):
            return object.__getattribute__(self, name)
        allowed = object.__getattribute__(self, "_allowed")
        if name not in allowed:
            raise DisallowedAttributeAccessError(
                f"Attempted to access undefined AgentResult attribute: '{name}'"
            )
        target = object.__getattribute__(self, "_target")
        return getattr(target, name)


# ---------------------------------------------------------------------------
# Tests: Real AgentResult Mapping and Tolerance
# ---------------------------------------------------------------------------


def test_agent_result_to_record_with_all_real_statuses(
    real_ok_result,
    real_refused_result,
    real_blocked_result,
    real_provider_unavailable_result,
):
    """agent_result_to_record must cleanly map real AgentResult instances across all 4 statuses."""
    # 1. OK Status
    rec_ok = agent_result_to_record(real_ok_result, elapsed_ms=1240.0)
    assert rec_ok["status"] == "ok"
    assert rec_ok["sql"] == real_ok_result.sql
    assert rec_ok["df"] is not None and len(rec_ok["df"]) == 1
    assert rec_ok["row_count"] == 1
    assert rec_ok["attempts"] == 1
    assert rec_ok["latency_ms"] == 1240.0
    assert rec_ok["is_demo"] is False
    msg_ok = format_status_message(rec_ok["status"], rec_ok["error_message"], rec_ok["explanation"])
    assert msg_ok["level"] == "success"

    # 2. Refused Status
    rec_ref = agent_result_to_record(real_refused_result, elapsed_ms=3.5)
    assert rec_ref["status"] == "refused"
    assert rec_ref["sql"] is None
    assert rec_ref["df"] is None
    assert rec_ref["row_count"] == 0
    assert "destructive" in rec_ref["explanation"]
    msg_ref = format_status_message(rec_ref["status"], rec_ref["error_message"], rec_ref["explanation"])
    assert msg_ref["level"] == "warning"
    assert "destructive" in msg_ref["message"]

    # 3. Blocked (Validator) Status
    rec_blk = agent_result_to_record(real_blocked_result, elapsed_ms=1520.0)
    assert rec_blk["status"] == "error"
    assert "denylisted column" in rec_blk["error_message"]
    msg_blk = format_status_message(rec_blk["status"], rec_blk["error_message"], rec_blk["explanation"])
    assert msg_blk["level"] == "error"
    assert "Security Validator" in msg_blk["title"]

    # 4. Provider Unavailable Status
    rec_pu = agent_result_to_record(real_provider_unavailable_result, elapsed_ms=42.0)
    assert rec_pu["status"] == "provider_unavailable"
    assert "exhausted" in rec_pu["error_message"]
    msg_pu = format_status_message(rec_pu["status"], rec_pu["error_message"], rec_pu["explanation"])
    assert msg_pu["level"] == "error"
    assert "Quota Reached" in msg_pu["title"]


def test_ui_never_reads_undefined_agent_result_attributes(
    real_ok_result,
    real_refused_result,
    real_blocked_result,
    real_provider_unavailable_result,
):
    """The UI adapter agent_result_to_record must never read attributes not defined on AgentResult.

    StrictAgentResultProxy raises DisallowedAttributeAccessError if any invalid attribute
    (such as 'refusal_reason') is accessed.
    """
    for res in [real_ok_result, real_refused_result, real_blocked_result, real_provider_unavailable_result]:
        proxy = StrictAgentResultProxy(res)
        # Must execute cleanly without DisallowedAttributeAccessError
        record = agent_result_to_record(proxy)
        assert record["status"] == res.status

    # Verify that StrictAgentResultProxy DOES raise on 'refusal_reason'
    proxy_test = StrictAgentResultProxy(real_refused_result)
    with pytest.raises(DisallowedAttributeAccessError, match="refusal_reason"):
        _ = proxy_test.refusal_reason


def test_ast_check_no_undefined_agent_result_attributes_in_ui_code():
    """Statically verify via AST that app/streamlit_app.py and app/ui_utils.py

    never access undefined attributes on agent_res.
    """
    valid_fields = set(AgentResult.model_fields.keys())

    for py_file in [Path("app/streamlit_app.py"), Path("app/ui_utils.py")]:
        assert py_file.exists(), f"File {py_file} must exist"
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            # Check direct attribute access: agent_res.<attr>
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in ("agent_res", "agent_result")
            ):
                assert node.attr in valid_fields, (
                    f"{py_file}:{node.lineno} accessed undefined AgentResult attribute '{node.attr}'"
                )
            # Check getattr(agent_res, "<attr>", ...)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in ("agent_res", "agent_result")
                and isinstance(node.args[1], ast.Constant)
            ):
                attr_name = node.args[1].value
                assert attr_name in valid_fields, (
                    f"{py_file}:{node.lineno} getattr called on undefined AgentResult attribute '{attr_name}'"
                )


# ---------------------------------------------------------------------------
# Tests: Streamlit AppTest Smoke Test
# ---------------------------------------------------------------------------


def test_streamlit_app_smoke_with_apptest():
    """Smoke test streamlit_app.py using AppTest with demo mode enabled.

    Clicks example chip 'ex_q01' ('Total Patients') and verifies no exceptions occur.
    Zero live API calls.
    """
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
    at = AppTest.from_file(app_path, default_timeout=25)
    at.session_state["demo_mode"] = True
    at.run()
    assert not at.exception

    # Click the ex_q01 button ('Total Patients')
    btn = at.button(key="ex_q01")
    assert btn is not None
    btn.click().run()
    assert not at.exception

    # Verify SQL block and table rendered
    assert len(at.code) > 0
    sql_text = at.code[0].value.lower()
    assert "count" in sql_text and "patients" in sql_text


# ---------------------------------------------------------------------------
# Tests: Example Questions and Demo Response Resolution
# ---------------------------------------------------------------------------


def test_example_questions_count_and_structure():
    """EXAMPLE_QUESTIONS must contain exactly 6 curated benchmark questions with required keys."""
    assert len(EXAMPLE_QUESTIONS) == 6
    for q in EXAMPLE_QUESTIONS:
        assert "id" in q
        assert "title" in q
        assert "question" in q
        assert "category" in q
        assert len(q["question"]) > 10


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


# ---------------------------------------------------------------------------
# Tests: Sanitization and Message Formatting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests: Session Quota Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests: Eval Summary Parsing (Verifying commit 7bbd679)
# ---------------------------------------------------------------------------


def test_load_eval_summary_from_latest_json():
    """load_eval_summary must dynamically parse metrics from eval/results/latest.json with commit 7bbd679."""
    summary = load_eval_summary()
    assert summary["model"] == "gemini-3.5-flash-lite"
    assert summary["git_commit"] == "7bbd679"
    assert summary["total_questions"] == 56
    assert summary["exact_accuracy_pct"] == 76.79
    assert summary["lenient_accuracy_pct"] == 96.43
    assert summary["safety_block_rate_pct"] == 100.0
    assert "Refused correctly" in summary["policy_refusal_status"]


def test_load_eval_summary_missing_file_fallback(tmp_path):
    """load_eval_summary must return graceful defaults if latest.json does not exist."""
    missing = tmp_path / "non_existent_results.json"
    summary = load_eval_summary(missing)
    assert isinstance(summary, dict)
    assert summary["git_commit"] == "7bbd679"
    assert summary["total_questions"] == 56


# ---------------------------------------------------------------------------
# Tests: Chart Selection & Integration
# ---------------------------------------------------------------------------


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
