"""Verification script that runs every gold SQL in eval/questions.yaml through the AST validator

and against the seeded clinic.db. Displays results before any eval model execution.
"""

import sys
from pathlib import Path

import yaml
from tabulate import tabulate

from text2sql.config import settings
from text2sql.executor import SQLExecutor
from text2sql.schema import introspect_schema
from text2sql.validator import SQLValidator


def verify_gold_queries(yaml_path: str = "eval/questions.yaml") -> bool:
    """Run every gold query in questions.yaml through SQLValidator and against the DB."""
    path = Path(yaml_path)
    if not path.exists():
        print(f"Error: {yaml_path} does not exist.")
        return False

    with open(path, encoding="utf-8") as f:
        questions = yaml.safe_load(f)

    print(f"Loaded {len(questions)} gold benchmark questions from {yaml_path}\n")

    executor = SQLExecutor(database_url=settings.database_url, timeout_seconds=10.0)
    schema = introspect_schema(executor.engine, denylist=settings.denylist_columns)
    validator = SQLValidator(
        allowed_tables=schema.table_names,
        allowed_columns=schema.get_allowed_columns(),
        denylist_columns=settings.denylist_columns,
        max_query_limit=settings.max_query_limit,
        reference_date=settings.reference_date,
    )

    table_data = []
    all_passed = True
    total_queries = len(questions)

    for q in questions:
        qid = q["id"]
        difficulty = q.get("difficulty", "unknown")
        category = q.get("category", "unknown")
        gold_sql = q["gold_sql"]

        # 1. AST Validation
        val_result = validator.validate(gold_sql)
        if not val_result.is_valid:
            all_passed = False
            table_data.append([
                qid,
                difficulty,
                category,
                "FAIL (Validator)",
                0,
                f"Validation error: {val_result.error_message}",
                gold_sql[:50] + "...",
            ])
            continue

        # 2. Database Execution
        try:
            exec_res = executor.execute(val_result.validated_sql)
            df = exec_res.df
            row_count = exec_res.row_count
            first_row_dict = df.iloc[0].to_dict() if row_count > 0 else {}
            # Format preview cleanly
            preview_items = [f"{k}={v}" for k, v in list(first_row_dict.items())[:3]]
            preview = ", ".join(preview_items) if preview_items else "(empty result set)"
            if len(first_row_dict) > 3:
                preview += ", ..."

            table_data.append([
                qid,
                difficulty,
                category,
                "PASS",
                row_count,
                preview[:60],
                gold_sql[:45] + "...",
            ])
        except Exception as e:
            all_passed = False
            table_data.append([
                qid,
                difficulty,
                category,
                "FAIL (DB Execution)",
                0,
                f"DB error: {e}",
                gold_sql[:50] + "...",
            ])

    headers = ["ID", "Difficulty", "Category", "Status", "Rows", "First Row Preview", "SQL (Truncated)"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))

    pass_count = sum(1 for row in table_data if row[3] == "PASS")
    print(f"\nVerification Summary: {pass_count}/{total_queries} queries PASSED.")

    if not all_passed:
        print("\nERROR: One or more gold SQL queries failed validation or execution.")
        return False

    print("\nSUCCESS: All gold queries validated and executed cleanly against clinic.db.")
    return True


if __name__ == "__main__":
    success = verify_gold_queries()
    sys.exit(0 if success else 1)
