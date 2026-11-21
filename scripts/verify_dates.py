"""Verification script to prove strftime() and date() work on stored timestamps."""

from sqlalchemy import create_engine, text

from text2sql.config import settings


def main():
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        print("--- Date and Timestamp Formatting Verification ---")

        # 1. Test functions on appointments.scheduled_at
        query = text("""
            SELECT 
                scheduled_at AS raw_str,
                date(scheduled_at) AS date_val,
                strftime('%Y', scheduled_at) AS year_val,
                strftime('%Y-%m', scheduled_at) AS year_month_val,
                strftime('%Y-%m-%d', scheduled_at) AS formatted_date,
                strftime('%H:%M', scheduled_at) AS time_val
            FROM appointments
            LIMIT 5
        """)
        rows = conn.execute(query).mappings().fetchall()
        for r in rows:
            print(dict(r))

        # 2. Confirm 0 nulls across all 30,000 appointments
        null_check = text("""
            SELECT 
                COUNT(*) AS total_rows,
                COUNT(scheduled_at) AS non_null_raw,
                SUM(CASE WHEN date(scheduled_at) IS NULL THEN 1 ELSE 0 END) AS null_dates,
                SUM(CASE WHEN strftime('%Y-%m', scheduled_at) IS NULL THEN 1 ELSE 0 END) AS null_strftimes
            FROM appointments
        """)
        null_res = conn.execute(null_check).mappings().fetchone()
        print("\n--- Integrity Across All Appointments ---")
        print(dict(null_res))
        assert null_res["null_dates"] == 0, "Found NULL dates!"
        assert null_res["null_strftimes"] == 0, "Found NULL strftime results!"

        # 3. Relative Date Filtering Proof
        rel_check = text("""
            SELECT COUNT(*) AS appts_2024
            FROM appointments
            WHERE date(scheduled_at) >= '2024-01-01' AND date(scheduled_at) <= '2024-12-31'
        """)
        appts_2024 = conn.execute(rel_check).scalar()
        print(f"\nTotal appointments in 2024 using date() comparison: {appts_2024:,}")
        print("Date/strftime validation: ALL TESTS PASSED.")


if __name__ == "__main__":
    main()
