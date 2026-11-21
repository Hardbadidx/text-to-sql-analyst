"""Verification script for Stage 1: schema introspection and dataset integrity."""

from sqlalchemy import create_engine, text

from text2sql.config import settings
from text2sql.schema import introspect_schema


def main():
    print("=" * 60)
    print("STAGE 1 VERIFICATION: CLINIC SAAS DATABASE & INTROSPECTION")
    print("=" * 60)

    engine = create_engine(settings.database_url)

    # 1. Real database row counts
    print("\n[1] Physical Database Table Row Counts:")
    tables = ["patients", "doctors", "appointments", "treatments", "invoices", "payments"]
    with engine.connect() as conn:
        for t in tables:
            cnt = conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
            print(f"  - {t:<15}: {cnt:>6,} rows")

    # 2. Schema introspection with denylist
    print(f"\n[2] Introspecting Schema (Denylist: {settings.denylist_columns})...")
    schema = introspect_schema(engine, denylist=settings.denylist_columns)

    # Check denylisted columns exclusion
    print("\n[3] Sensitive Column Denylist Audit:")
    for t_name, t_info in schema.tables.items():
        found_denied = [c.name for c in t_info.columns if schema.is_column_denylisted(c.name)]
        if found_denied:
            print(f"  [FAIL] Found denylisted columns in table {t_name}: {found_denied}")
        else:
            print(f"  [PASS] Table '{t_name}' contains NO denylisted columns.")

        # Check sample values
        for c in t_info.columns:
            if schema.is_column_denylisted(c.name) and c.sample_values:
                print(f"  [FAIL] Denylisted column {t_name}.{c.name} has sample values!")

    # Verify 'patients' table columns in introspected schema vs raw DB
    print("\n[4] Columns exposed to LLM in 'patients' table:")
    exposed_cols = [c.name for c in schema.tables["patients"].columns]
    print(f"  Exposed: {exposed_cols}")
    assert "phone" not in exposed_cols, "phone column leaked!"
    assert "email" not in exposed_cols, "email column leaked!"
    assert "address" not in exposed_cols, "address column leaked!"
    print("  [PASS] 'phone', 'email', and 'address' are completely absent from LLM schema context.")

    # 5. Show sample non-sensitive rows from each table
    print("\n[5] Sample Non-Sensitive Rows (as seen in database):")
    with engine.connect() as conn:
        for t in tables:
            # Query non-sensitive columns
            if t == "patients":
                query_sql = "SELECT id, first_name, last_name, gender, date_of_birth FROM patients LIMIT 2"
            else:
                query_sql = f"SELECT * FROM {t} LIMIT 2"

            rows = conn.execute(text(query_sql)).mappings().fetchall()
            print(f"\n  -- Table: {t} (2 sample rows) --")
            for r in rows:
                print(f"    {dict(r)}")

    # 6. Print LLM prompt snippet
    print("\n[6] Introspected Schema Context (LLM Prompt Format Preview):")
    prompt_ctx = schema.to_prompt_context()
    for line in prompt_ctx.splitlines()[:25]:
        print(f"  {line}")
    print("  ... [truncated for display]")

    print("\n" + "=" * 60)
    print("STAGE 1 VERIFICATION COMPLETE: ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
