"""Build the demo replay cache (app/demo_cache.json) from real eval run records and clinic.db.

Takes real model-generated SQL and true execution latencies from eval/results/latest.json
(or eval/cache), re-executes each query against clinic.db through the AST validator and
read-only executor, and stores the verified results, true latencies, and chart recommendations.

Zero live API calls. Zero hard-coded/fabricated results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.ui_utils import EXAMPLE_QUESTIONS
from text2sql.agent import Text2SQLAgent
from text2sql.charts import auto_pick_chart_type


def build_demo_cache(
    output_path: Path | str = "app/demo_cache.json",
    latest_json_path: Path | str = "eval/results/latest.json",
    db_path: Path | str = "clinic.db",
) -> dict[str, dict[str, Any]]:
    """Build and save the demo cache JSON file from real benchmark run data."""
    out_file = Path(output_path)
    latest_file = Path(latest_json_path)
    database_file = Path(db_path)

    if not latest_file.exists():
        raise FileNotFoundError(
            f"Eval results record not found at {latest_file.resolve()}. "
            "A real eval run record is required to build the demo cache."
        )

    if not database_file.exists():
        raise FileNotFoundError(
            f"Clinic database not found at {database_file.resolve()}."
        )

    with open(latest_file, encoding="utf-8") as f:
        latest_data = json.load(f)

    # Index real eval results by question id and by normalized question text
    results_by_id = {item["id"]: item for item in latest_data.get("results", []) if "id" in item}
    results_by_q = {
        item["question"].strip().lower(): item
        for item in latest_data.get("results", [])
        if "question" in item
    }

    # Initialize agent for real AST validation and read-only DB execution
    agent = Text2SQLAgent()
    demo_cache: dict[str, dict[str, Any]] = {}

    print(f"Building demo cache from {latest_file} against {database_file}...")

    for ex in EXAMPLE_QUESTIONS:
        qid = ex["id"]
        q_text = ex["question"]
        norm_q = q_text.strip().lower()

        # Retrieve real run record
        item = results_by_id.get(qid) or results_by_q.get(norm_q)
        if not item:
            raise RuntimeError(
                f"Question {qid} ('{q_text}') has no real cached run in {latest_file}. Stopping."
            )

        status = item.get("status")
        sql = item.get("sql")
        latency_ms = item.get("latency_ms")
        attempts = item.get("attempts", 1)

        if status != "ok" or not sql:
            raise RuntimeError(
                f"Question {qid} does not have a successful real cached run (status={status}, sql={sql}). Stopping."
            )

        if latency_ms is None:
            raise RuntimeError(
                f"Question {qid} has no real recorded latency in {latest_file}. Stopping."
            )

        # Re-execute the SQL against clinic.db through validator and executor
        v_res = agent.validator.validate(sql)
        if not v_res.is_valid:
            raise RuntimeError(
                f"Validator rejected real cached SQL for {qid}: {v_res.error_message}\nSQL: {sql}"
            )

        exec_res = agent.executor.execute(v_res.validated_sql)
        df = exec_res.df
        rows = df.to_dict(orient="records")
        chart_type = auto_pick_chart_type(df)

        entry = {
            "question_id": qid,
            "title": ex["title"],
            "category": ex["category"],
            "question": q_text,
            "status": "ok",
            "sql": sql,
            "data": rows,
            "row_count": len(rows),
            "latency_ms": round(float(latency_ms), 2),
            "attempts": int(attempts),
            "chart_type": chart_type,
            "explanation": f"Re-executed from verified benchmark SQL against {database_file.name}.",
        }

        # Index by normalized question text (primary lookup key)
        demo_cache[norm_q] = entry

        print(
            f"  [{qid}] '{ex['title']}': {len(rows)} row(s), "
            f"latency={entry['latency_ms']} ms, chart={chart_type}"
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(demo_cache, f, indent=2)

    print(f"\nSuccessfully wrote {len(demo_cache)} verified entries to {out_file.resolve()}.")
    return demo_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build app/demo_cache.json from real eval records and clinic.db.")
    parser.add_argument("--output", default="app/demo_cache.json", help="Output JSON path")
    parser.add_argument("--latest-json", default="eval/results/latest.json", help="Eval results JSON")
    parser.add_argument("--db-path", default="clinic.db", help="Path to SQLite database")
    args = parser.parse_args()

    try:
        build_demo_cache(
            output_path=args.output,
            latest_json_path=args.latest_json,
            db_path=args.db_path,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
