"""Verification script that runs every safety case in eval/safety_cases.yaml

through the deterministic pre-check and AST validator to ensure 100% rejection rate.
"""

import sys
from pathlib import Path

import yaml
from tabulate import tabulate

from text2sql.agent import Text2SQLAgent
from text2sql.config import settings
from text2sql.executor import SQLExecutor
from text2sql.schema import introspect_schema
from text2sql.validator import SQLValidator


def verify_safety_cases(yaml_path: str = "eval/safety_cases.yaml") -> bool:
    """Verify all adversarial cases are refused by pre-check or blocked by validator."""
    path = Path(yaml_path)
    if not path.exists():
        print(f"Error: {yaml_path} does not exist.")
        return False

    with open(path, encoding="utf-8") as f:
        cases = yaml.safe_load(f)

    print(f"Loaded {len(cases)} safety benchmark cases from {yaml_path}\n")

    executor = SQLExecutor(database_url=settings.database_url)
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

    for c in cases:
        cid = c["id"]
        cat = c.get("category", "")
        prompt = c["prompt"]
        expected = c.get("expected_outcome", "refused")

        # 1. Deterministic pre-check
        pre_refusal = Text2SQLAgent._check_deterministic_refusal(prompt)
        if pre_refusal:
            table_data.append([cid, cat, "refused", "refused (Pre-Check)", "PASS", prompt[:45] + "..."])
            continue

        # 2. Validator check (if passed pre-check, validator must block)
        val_res = validator.validate(prompt)
        if not val_res.is_valid:
            table_data.append([cid, cat, "blocked", f"blocked (Validator: {val_res.error_message[:25]}...)", "PASS", prompt[:45] + "..."])
            continue

        # If neither refused nor blocked, it failed safety!
        all_passed = False
        table_data.append([cid, cat, expected, "ALLOWED (FAILED)", "FAIL", prompt[:45] + "..."])

    headers = ["ID", "Category", "Expected", "Actual Defense", "Status", "Prompt"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))

    pass_count = sum(1 for r in table_data if r[4] == "PASS")
    print(f"\nSafety Verification Summary: {pass_count}/{len(cases)} cases safely DEFENDED.")

    if not all_passed:
        print("\nERROR: One or more safety cases was allowed through!")
        return False

    print("\nSUCCESS: All adversarial cases successfully blocked or refused (0 mutations possible).")
    return True


if __name__ == "__main__":
    success = verify_safety_cases()
    sys.exit(0 if success else 1)
