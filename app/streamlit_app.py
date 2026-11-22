"""Streamlit Web Application for Text-to-SQL Clinical Analyst.

Features:
- Natural language query box with 6 benchmark example question chips.
- SQL code viewer, interactive result table, and auto-picked Plotly visualizations.
- AST security validation and read-only execution boundary.
- Human-readable status messages (refusal, validator block, quota limit, provider error).
- Sidebar with non-denylisted schema viewer, reference date, synthetic data note, and dynamic eval summary.
- Session history of previous questions, clickable to re-inspect results.
- Session quota protection (max 10 live queries) with automatic fallback to pre-verified Demo Mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for non-editable installations and container runtimes
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
from typing import Any

import pandas as pd
import streamlit as st

from app.ui_utils import (
    EXAMPLE_QUESTIONS,
    agent_result_to_record,
    check_quota_limit,
    format_status_message,
    get_demo_response,
    get_remaining_quota,
    load_eval_summary,
)
from text2sql.agent import AgentResult, Text2SQLAgent
from text2sql.charts import auto_pick_chart_type, generate_plotly_figure

MAX_LIVE_QUERIES = 10

# Configure page layout
st.set_page_config(
    page_title="Text-to-SQL Clinical Analyst",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_agent() -> Text2SQLAgent:
    """Initialize and cache the Text2SQLAgent instance."""
    return Text2SQLAgent()


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "live_query_count" not in st.session_state:
        st.session_state["live_query_count"] = 0
    if "demo_mode" not in st.session_state:
        st.session_state["demo_mode"] = False
    if "active_result" not in st.session_state:
        st.session_state["active_result"] = None
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = ""


def render_sidebar(agent: Text2SQLAgent, eval_summary: dict[str, Any]) -> None:
    """Render the sidebar containing database schema, eval metrics, and session history."""
    with st.sidebar:
        st.title("🏥 Clinic Data Explorer")
        st.caption("Safety-First Clinical Text-to-SQL Assistant")

        # 1. Quota & Demo Mode Status
        st.subheader("🛡️ Demo & Quota Protection")
        remaining = get_remaining_quota(st.session_state["live_query_count"], MAX_LIVE_QUERIES)

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.metric("Live Queries Left", f"{remaining}/{MAX_LIVE_QUERIES}")
        with col_q2:
            demo_active = st.session_state["demo_mode"] or (remaining == 0)
            st.metric("Mode", "Demo" if demo_active else "Live LLM")

        # Demo mode toggle directly bound to session state
        st.toggle("Enable Demo Mode (Offline / Cached)", key="demo_mode")

        st.divider()

        # 2. Database Schema Viewer (Strictly Non-Denylisted Columns Only)
        st.subheader("📋 Database Schema")
        st.caption("🔒 Sensitive contact PII (phone, email, address, ssn) is omitted by policy.")

        schema = agent.schema
        for tbl_name, tbl_info in schema.tables.items():
            row_count_str = f" (~{tbl_info.row_count:,} rows)" if tbl_info.row_count is not None else ""
            with st.expander(f"📁 {tbl_name}{row_count_str}"):
                cols_data = []
                for col in tbl_info.columns:
                    col_type = col.type_name
                    if col.primary_key:
                        col_type += " 🔑 [PK]"
                    cols_data.append({"Column": col.name, "Type": col_type})
                st.dataframe(pd.DataFrame(cols_data), hide_index=True, use_container_width=True)

        st.divider()

        # 3. System Configuration & Data Notice
        st.subheader("⚙️ System Reference")
        st.info(f"**Reference Date**: `{agent.validator.reference_date}`\n*(All relative date queries align to this date)*")
        st.caption("ℹ️ **Dataset Notice**: This database consists of synthetic clinic SaaS records generated for analytical benchmarking.")

        st.divider()

        # 4. Evaluation Benchmark Summary (Dynamically read from latest.json)
        st.subheader("📊 Benchmark Results")
        if not eval_summary.get("available", False):
            st.caption("eval results unavailable")
        else:
            st.caption(f"Evaluated on {eval_summary['model']} (Commit: `{eval_summary['git_commit']}`)")
            st.write(f"- **Exact Match Accuracy**: `{eval_summary['exact_accuracy_pct']}%`")
            st.write(f"- **Lenient Accuracy**: `{eval_summary['lenient_accuracy_pct']}%`")
            st.write(f"- **Safety Block Rate**: `{eval_summary['safety_block_rate_pct']}%`")
            st.write(f"- **Policy Refusal**: `{eval_summary['policy_refusal_status']}`")
            st.write(f"- **Mean Latency**: `{eval_summary['mean_latency_ms']:.1f} ms`")

        st.divider()

        # 5. Session History
        st.subheader("🕒 Session History")
        if not st.session_state["history"]:
            st.caption("No queries run in this session yet.")
        else:
            for idx, hist_entry in enumerate(reversed(st.session_state["history"])):
                q_text = hist_entry.get("question", "")
                status = hist_entry.get("status", "ok")
                icon = "✅" if status == "ok" else ("🛡️" if status == "refused" else "🚫")
                btn_label = f"{icon} {q_text[:35]}..." if len(q_text) > 35 else f"{icon} {q_text}"
                if st.button(btn_label, key=f"hist_{idx}"):
                    st.session_state["active_result"] = hist_entry
                    st.session_state["current_question"] = q_text

            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["history"] = []
                st.session_state["active_result"] = None


def run_query(question: str, agent: Text2SQLAgent) -> None:
    """Execute query through Text2SQLAgent or fallback to Demo Mode if quota is reached."""
    cleaned_q = question.strip()
    if not cleaned_q:
        return

    # Check if demo mode or session limit reached
    is_quota_exhausted = check_quota_limit(st.session_state["live_query_count"], MAX_LIVE_QUERIES)
    use_demo = st.session_state["demo_mode"] or is_quota_exhausted

    if use_demo:
        demo_resp = get_demo_response(cleaned_q)
        if demo_resp:
            record: dict[str, Any] = {
                "question": cleaned_q,
                "status": demo_resp.get("status", "ok"),
                "sql": demo_resp.get("sql"),
                "df": demo_resp.get("df"),
                "row_count": len(demo_resp["df"]) if demo_resp.get("df") is not None else 0,
                "explanation": demo_resp.get("explanation", "Pre-verified benchmark response from demonstration cache."),
                "attempts": demo_resp.get("attempts", 1),
                "latency_ms": demo_resp.get("latency_ms", 0.0),
                "is_demo": True,
                "error_message": demo_resp.get("error_message"),
                "debug_error": None,
                "chart_type": demo_resp.get("chart_type", "table-only"),
                "chart_figure": None,
            }
            st.session_state["active_result"] = record
            st.session_state["history"].append(record)
            return

        if is_quota_exhausted:
            # Question is not an example and live limit is reached
            record = {
                "question": cleaned_q,
                "status": "refused",
                "sql": None,
                "df": None,
                "row_count": 0,
                "explanation": (
                    f"Session quota limit reached ({MAX_LIVE_QUERIES} live questions per session). "
                    "Please select one of the 6 example questions above to explore pre-verified demo results."
                ),
                "attempts": 0,
                "latency_ms": 0.0,
                "is_demo": True,
                "error_message": "Live query limit reached for this session.",
                "debug_error": None,
                "chart_type": "table-only",
                "chart_figure": None,
            }
        else:
            # Demo mode active but query is not in demo_cache.json
            record = {
                "question": cleaned_q,
                "status": "error",
                "sql": None,
                "df": None,
                "row_count": 0,
                "explanation": (
                    "Demo data unavailable for this question. "
                    "Only the 6 curated benchmark examples have pre-verified responses in Demo Mode."
                ),
                "attempts": 0,
                "latency_ms": 0.0,
                "is_demo": True,
                "error_message": "Demo data unavailable: No pre-verified benchmark response exists in demo_cache.json.",
                "debug_error": None,
                "chart_type": "table-only",
                "chart_figure": None,
            }
        st.session_state["active_result"] = record
        st.session_state["history"].append(record)
        return

    # Live Execution via Text2SQLAgent
    with st.spinner("Analyzing schema, generating SQL, and verifying AST constraints..."):
        start_t = time.perf_counter()
        agent_res: AgentResult = agent.ask(cleaned_q)
        elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

        # Detect quota exhaustion from provider
        status = getattr(agent_res, "status", "ok")
        err_msg = getattr(agent_res, "error_message", None)
        if status == "provider_unavailable":
            err_lower = (err_msg or "").lower()
            if any(w in err_lower for w in ["quota", "perday", "429", "resource_exhausted"]):
                st.session_state["demo_mode"] = True

        st.session_state["live_query_count"] += 1

        record = agent_result_to_record(agent_res, elapsed_ms)
        st.session_state["active_result"] = record
        st.session_state["history"].append(record)


def render_result_view(result: dict[str, Any] | Any) -> None:
    """Render the active query result including banner, metrics, SQL, dataframe, and charts.

    Tolerant to both dictionaries and object instances via getattr.
    """
    status = getattr(result, "status", None) if not isinstance(result, dict) else result.get("status", "ok")
    err_msg = getattr(result, "error_message", None) if not isinstance(result, dict) else result.get("error_message")
    explanation = getattr(result, "explanation", None) if not isinstance(result, dict) else result.get("explanation")
    refusal_reason = getattr(result, "refusal_reason", None) if not isinstance(result, dict) else result.get("refusal_reason")
    is_demo = getattr(result, "is_demo", False) if not isinstance(result, dict) else result.get("is_demo", False)

    # 1. Demo Mode Notice
    if is_demo:
        st.info("💡 **Demo Mode Replay**: Serving pre-verified benchmark response from the demonstration cache.")

    # 2. Human-Readable Status Banner
    status_info = format_status_message(status or "ok", err_msg, explanation, refusal_reason)
    if status_info["level"] == "warning":
        st.warning(f"🛡️ **{status_info['title']}**: {status_info['message']}")
    elif status_info["level"] == "error":
        st.error(f"🚫 **{status_info['title']}**: {status_info['message']}")
    elif status_info["level"] == "info":
        st.info(f"ℹ️ **{status_info['title']}**: {status_info['message']}")

    df = getattr(result, "df", None) if not isinstance(result, dict) else result.get("df")
    sql = getattr(result, "sql", None) if not isinstance(result, dict) else result.get("sql")
    latency_ms = getattr(result, "latency_ms", 0.0) if not isinstance(result, dict) else result.get("latency_ms", 0.0)
    attempts = getattr(result, "attempts", 1) if not isinstance(result, dict) else result.get("attempts", 1)
    chart_type = getattr(result, "chart_type", None) if not isinstance(result, dict) else result.get("chart_type")
    chart_figure = getattr(result, "chart_figure", None) if not isinstance(result, dict) else result.get("chart_figure")
    question = getattr(result, "question", "") if not isinstance(result, dict) else result.get("question", "")

    # 3. Execution Metrics (Only for valid results)
    if status == "ok" and df is not None:
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            latency_label = "Original Latency" if is_demo else "Latency"
            st.metric(latency_label, f"{latency_ms:.0f} ms")
        with col_m2:
            st.metric("Attempts", f"{attempts} / 3")
        with col_m3:
            st.metric("Rows Returned", f"{len(df):,}")

        # 4. Generated SQL Code Block
        st.markdown("### 📝 Generated SQL Query")
        st.code(sql or "-- No query generated", language="sql")

        # 5. Visualizations & Data Tabs
        st.markdown("### 📊 Query Results & Visualizations")
        tab_chart, tab_data = st.tabs(["Interactive Visualization", "Data Table"])

        with tab_data:
            st.dataframe(df, use_container_width=True)

        with tab_chart:
            c_type = chart_type or auto_pick_chart_type(df)
            if c_type == "table-only" or len(df) == 0:
                st.caption("ℹ️ Tabular presentation is best for this single-value or empty result set.")
                st.dataframe(df, use_container_width=True)
            else:
                fig = chart_figure or generate_plotly_figure(df, chart_type=c_type, title=question)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)

    # 6. Plain-English Explanation
    if explanation:
        st.markdown("### 💬 Analyst Explanation")
        st.markdown(explanation)


def main() -> None:
    """Main Streamlit application entrypoint."""
    init_session_state()
    agent = get_agent()
    eval_summary = load_eval_summary()

    render_sidebar(agent, eval_summary)

    # Main Application Header
    st.title("🏥 Text-to-SQL Clinical Analyst")
    st.markdown(
        "Ask natural-language questions across clinic appointments, patient encounters, doctor specialties, and billing. "
        "Every generated query is strictly verified by an **AST security validator** and executed on a **read-only database**."
    )

    if st.session_state["demo_mode"]:
        st.warning(
            "⚠️ **Demo Mode Active**: Live LLM calls are disabled. "
            "You can click and explore any of the 6 benchmark questions below to view pre-verified SQL, tables, and charts."
        )

    st.markdown("#### ⚡ Example Benchmark Questions")
    st.caption("Click any example to load and execute:")

    # 6 Example Question Chips arranged in two rows of 3 columns
    row1_cols = st.columns(3)
    row2_cols = st.columns(3)

    for i, ex in enumerate(EXAMPLE_QUESTIONS):
        target_col = row1_cols[i] if i < 3 else row2_cols[i - 3]
        with target_col:
            if st.button(f"{ex['icon']} {ex['title']}", key=f"ex_{ex['id']}", use_container_width=True):
                st.session_state["current_question"] = ex["question"]
                run_query(ex["question"], agent)

    st.divider()

    # Question Input Form
    with st.form("query_form", clear_on_submit=False):
        user_question = st.text_input(
            "Enter your analysis question:",
            value=st.session_state.get("current_question", ""),
            placeholder="e.g., What is the total invoiced revenue by specialty in 2024?",
        )
        submit_btn = st.form_submit_button("🚀 Run Query", type="primary", use_container_width=True)

    if submit_btn and user_question.strip():
        st.session_state["current_question"] = user_question.strip()
        run_query(user_question.strip(), agent)

    # Display Active Result
    if st.session_state["active_result"]:
        st.divider()
        render_result_view(st.session_state["active_result"])


if __name__ == "__main__":
    main()
