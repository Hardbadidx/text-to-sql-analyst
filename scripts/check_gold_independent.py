"""Independent verification and spot-check of 10 benchmark queries.

Formulates completely new, independent SQL queries (using different constructs,
CTEs, subqueries, and aggregation functions) and runs both side-by-side against
clinic.db in read-only mode to verify the accuracy of the gold benchmark queries.

Includes every ratio/percentage question (q30, q31, q33, q36, q40, q43, q54),
plus q18, q21, and q42.
"""

import sqlite3
import sys
from pathlib import Path

from tabulate import tabulate

DB_PATH = Path("clinic.db")

VERIFICATION_CASES = [
    {
        "id": "q18",
        "question": (
            "List the top 5 doctors with the highest number of completed appointments, "
            "breaking any ties by doctor last name ascending, then first name ascending."
        ),
        "gold_sql": (
            "SELECT d.first_name, d.last_name, COUNT(a.id) AS completed_count "
            "FROM doctors d JOIN appointments a ON d.id = a.doctor_id "
            "WHERE a.status = 'completed' "
            "GROUP BY d.id, d.first_name, d.last_name "
            "ORDER BY completed_count DESC, d.last_name ASC, d.first_name ASC "
            "LIMIT 5"
        ),
        "independent_sql": (
            "WITH doc_completed AS ("
            "    SELECT doctor_id, COUNT(*) AS completed_count "
            "    FROM appointments "
            "    WHERE status = 'completed' "
            "    GROUP BY doctor_id"
            ") "
            "SELECT d.first_name, d.last_name, dc.completed_count "
            "FROM doc_completed dc "
            "JOIN doctors d ON dc.doctor_id = d.id "
            "ORDER BY dc.completed_count DESC, d.last_name ASC, d.first_name ASC "
            "LIMIT 5"
        ),
    },
    {
        "id": "q21",
        "question": (
            "What is the monthly invoiced revenue for each month in 2024, ordered by month ascending?"
        ),
        "gold_sql": (
            "SELECT strftime('%Y-%m', issued_at) AS month, ROUND(SUM(amount), 2) AS monthly_invoiced "
            "FROM invoices "
            "WHERE issued_at >= '2024-01-01' AND issued_at < '2025-01-01' "
            "GROUP BY month "
            "ORDER BY month ASC"
        ),
        "independent_sql": (
            "SELECT substr(issued_at, 1, 7) AS month, ROUND(TOTAL(amount), 2) AS monthly_invoiced "
            "FROM invoices "
            "WHERE strftime('%Y', issued_at) = '2024' "
            "GROUP BY substr(issued_at, 1, 7) "
            "ORDER BY month ASC"
        ),
    },
    {
        "id": "q30",
        "question": (
            "What is the cancellation rate of appointments in 2024 as a percentage rounded to 2 decimals?"
        ),
        "gold_sql": (
            "SELECT ROUND(100.0 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancellation_rate_pct "
            "FROM appointments "
            "WHERE scheduled_at >= '2024-01-01' AND scheduled_at < '2025-01-01'"
        ),
        "independent_sql": (
            "WITH appt_2024 AS ("
            "    SELECT status FROM appointments WHERE strftime('%Y', scheduled_at) = '2024'"
            ") "
            "SELECT ROUND(100.0 * (SELECT COUNT(*) FROM appt_2024 WHERE status = 'cancelled') / (SELECT COUNT(*) FROM appt_2024), 2) AS cancellation_rate_pct"
        ),
    },
    {
        "id": "q31",
        "question": (
            "What is the no-show rate by doctor as a percentage rounded to 2 decimals, showing doctors "
            "with at least 50 total appointments, ordered by highest no-show rate descending, then total appointments descending?"
        ),
        "gold_sql": (
            "SELECT d.first_name, d.last_name, ROUND(100.0 * SUM(CASE WHEN a.status = 'no_show' THEN 1 ELSE 0 END) / COUNT(a.id), 2) AS no_show_rate_pct "
            "FROM doctors d JOIN appointments a ON d.id = a.doctor_id "
            "GROUP BY d.id, d.first_name, d.last_name "
            "HAVING COUNT(a.id) >= 50 "
            "ORDER BY no_show_rate_pct DESC, COUNT(a.id) DESC"
        ),
        "independent_sql": (
            "WITH doc_stats AS ("
            "    SELECT doctor_id, COUNT(*) AS total_appts, TOTAL(status = 'no_show') AS no_shows "
            "    FROM appointments "
            "    GROUP BY doctor_id "
            "    HAVING COUNT(*) >= 50"
            ") "
            "SELECT d.first_name, d.last_name, ROUND(100.0 * ds.no_shows / ds.total_appts, 2) AS no_show_rate_pct "
            "FROM doc_stats ds "
            "JOIN doctors d ON ds.doctor_id = d.id "
            "ORDER BY no_show_rate_pct DESC, ds.total_appts DESC"
        ),
    },
    {
        "id": "q33",
        "question": (
            "What is the payment collection efficiency ratio (total payments amount divided by total invoiced amount) "
            "per specialty as a percentage rounded to 2 decimals, ordered by collection rate descending?"
        ),
        "gold_sql": (
            "WITH specialty_invoices AS ("
            "    SELECT d.specialty, SUM(i.amount) AS total_invoiced "
            "    FROM doctors d JOIN appointments a ON d.id = a.doctor_id "
            "    JOIN invoices i ON a.id = i.appointment_id "
            "    GROUP BY d.specialty"
            "), specialty_payments AS ("
            "    SELECT d.specialty, SUM(p.amount) AS total_paid "
            "    FROM doctors d JOIN appointments a ON d.id = a.doctor_id "
            "    JOIN invoices i ON a.id = i.appointment_id "
            "    JOIN payments p ON i.id = p.invoice_id "
            "    GROUP BY d.specialty"
            ") "
            "SELECT si.specialty, ROUND(100.0 * COALESCE(sp.total_paid, 0) / si.total_invoiced, 2) AS collection_rate_pct "
            "FROM specialty_invoices si "
            "LEFT JOIN specialty_payments sp ON si.specialty = sp.specialty "
            "ORDER BY collection_rate_pct DESC"
        ),
        "independent_sql": (
            "SELECT d.specialty, "
            "       ROUND(100.0 * IFNULL(("
            "           SELECT SUM(p.amount) "
            "           FROM payments p "
            "           JOIN invoices i2 ON p.invoice_id = i2.id "
            "           JOIN appointments a2 ON i2.appointment_id = a2.id "
            "           JOIN doctors d2 ON a2.doctor_id = d2.id "
            "           WHERE d2.specialty = d.specialty"
            "       ), 0) / SUM(i.amount), 2) AS collection_rate_pct "
            "FROM doctors d "
            "JOIN appointments a ON d.id = a.doctor_id "
            "JOIN invoices i ON a.id = i.appointment_id "
            "GROUP BY d.specialty "
            "ORDER BY collection_rate_pct DESC"
        ),
    },
    {
        "id": "q36",
        "question": (
            "Which specialties have a higher cancellation rate than the clinic-wide average cancellation rate, "
            "showing their cancellation rate as a percentage rounded to 2 decimals, ordered from highest to lowest?"
        ),
        "gold_sql": (
            "WITH spec_rates AS ("
            "    SELECT d.specialty, 100.0 * SUM(CASE WHEN a.status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(a.id) AS cancel_rate "
            "    FROM doctors d JOIN appointments a ON d.id = a.doctor_id "
            "    GROUP BY d.specialty"
            ") "
            "SELECT specialty, ROUND(cancel_rate, 2) AS cancel_rate_pct "
            "FROM spec_rates "
            "WHERE cancel_rate > ("
            "    SELECT 100.0 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*) FROM appointments"
            ") "
            "ORDER BY cancel_rate_pct DESC"
        ),
        "independent_sql": (
            "SELECT d.specialty, ROUND(AVG(a.status = 'cancelled') * 100.0, 2) AS cancel_rate_pct "
            "FROM doctors d "
            "JOIN appointments a ON d.id = a.doctor_id "
            "GROUP BY d.specialty "
            "HAVING (AVG(a.status = 'cancelled') * 100.0) > ("
            "    SELECT AVG(status = 'cancelled') * 100.0 FROM appointments"
            ") "
            "ORDER BY cancel_rate_pct DESC"
        ),
    },
    {
        "id": "q40",
        "question": (
            "Find all patients who had at least 10 appointments in 2024 and calculate their completed "
            "appointment ratio as a percentage rounded to 2 decimals, ordered by total appointments descending, "
            "completion percentage descending, then patient last name ascending."
        ),
        "gold_sql": (
            "SELECT p.first_name, p.last_name, ROUND(100.0 * SUM(CASE WHEN a.status = 'completed' THEN 1 ELSE 0 END) / COUNT(a.id), 2) AS completion_pct "
            "FROM patients p JOIN appointments a ON p.id = a.patient_id "
            "WHERE a.scheduled_at >= '2024-01-01' AND a.scheduled_at < '2025-01-01' "
            "GROUP BY p.id, p.first_name, p.last_name "
            "HAVING COUNT(a.id) >= 10 "
            "ORDER BY COUNT(a.id) DESC, completion_pct DESC, p.last_name ASC"
        ),
        "independent_sql": (
            "WITH patient_2024_appts AS ("
            "    SELECT patient_id, COUNT(*) AS total_appts, TOTAL(status = 'completed') AS completed_count "
            "    FROM appointments "
            "    WHERE strftime('%Y', scheduled_at) = '2024' "
            "    GROUP BY patient_id "
            "    HAVING COUNT(*) >= 10"
            ") "
            "SELECT p.first_name, p.last_name, ROUND(100.0 * pta.completed_count / pta.total_appts, 2) AS completion_pct "
            "FROM patient_2024_appts pta "
            "JOIN patients p ON pta.patient_id = p.id "
            "ORDER BY pta.total_appts DESC, completion_pct DESC, p.last_name ASC"
        ),
    },
    {
        "id": "q42",
        "question": (
            "What is the monthly growth in collected payment revenue across 2024, ordered by month ascending?"
        ),
        "gold_sql": (
            "WITH monthly_rev AS ("
            "    SELECT strftime('%Y-%m', payment_date) AS month, ROUND(SUM(amount), 2) AS revenue "
            "    FROM payments "
            "    WHERE payment_date >= '2024-01-01' AND payment_date < '2025-01-01' "
            "    GROUP BY month"
            ") "
            "SELECT month, ROUND(revenue - LAG(revenue, 1) OVER (ORDER BY month), 2) AS mom_difference "
            "FROM monthly_rev "
            "ORDER BY month ASC"
        ),
        "independent_sql": (
            "WITH rev_by_month AS ("
            "    SELECT substr(payment_date, 1, 7) AS month, ROUND(SUM(amount), 2) AS revenue "
            "    FROM payments "
            "    WHERE payment_date >= '2024-01-01 00:00:00' AND payment_date <= '2024-12-31 23:59:59' "
            "    GROUP BY substr(payment_date, 1, 7)"
            ") "
            "SELECT curr.month, ROUND(curr.revenue - prev.revenue, 2) AS mom_difference "
            "FROM rev_by_month curr "
            "LEFT JOIN rev_by_month prev ON prev.month = strftime('%Y-%m', date(curr.month || '-01', '-1 month')) "
            "ORDER BY curr.month ASC"
        ),
    },
    {
        "id": "q43",
        "question": (
            "What is the appointment drop-off rate between scheduled appointments and completed appointments "
            "in 2024 as a percentage rounded to 2 decimals?"
        ),
        "gold_sql": (
            "SELECT ROUND(100.0 * SUM(CASE WHEN status != 'completed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS drop_off_rate_pct "
            "FROM appointments "
            "WHERE scheduled_at >= '2024-01-01' AND scheduled_at < '2025-01-01'"
        ),
        "independent_sql": (
            "SELECT ROUND(100.0 - (100.0 * TOTAL(status = 'completed') / COUNT(*)), 2) AS drop_off_rate_pct "
            "FROM appointments "
            "WHERE strftime('%Y', scheduled_at) = '2024'"
        ),
    },
    {
        "id": "q54",
        "question": (
            "Update me on the no-show rate in 2024 as a percentage rounded to 2 decimals"
        ),
        "gold_sql": (
            "SELECT ROUND(100.0 * SUM(CASE WHEN status = 'no_show' THEN 1 ELSE 0 END) / COUNT(*), 2) AS no_show_rate_pct "
            "FROM appointments "
            "WHERE scheduled_at >= '2024-01-01' AND scheduled_at < '2025-01-01'"
        ),
        "independent_sql": (
            "SELECT ROUND(100.0 * TOTAL(status = 'no_show') / COUNT(*), 2) AS no_show_rate_pct "
            "FROM appointments "
            "WHERE strftime('%Y', scheduled_at) = '2024'"
        ),
    },
]


def run_spot_check() -> bool:
    if not DB_PATH.exists():
        print(f"ERROR: Database file '{DB_PATH}' not found.")
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    all_passed = True
    summary_rows = []

    print("=" * 80)
    print("INDEPENDENT SPOT-CHECK OF 10 BENCHMARK GOLD QUERIES (clinic.db read-only)")
    print("=" * 80)

    try:
        for case in VERIFICATION_CASES:
            qid = case["id"]
            gold_sql = case["gold_sql"]
            indep_sql = case["independent_sql"]

            cursor_gold = conn.execute(gold_sql)
            gold_rows = cursor_gold.fetchall()

            cursor_indep = conn.execute(indep_sql)
            indep_rows = cursor_indep.fetchall()

            match = gold_rows == indep_rows
            if not match:
                all_passed = False

            sample_display = str(gold_rows[0]) if gold_rows else "EMPTY"
            if len(gold_rows) > 1:
                sample_display += f" ... (+{len(gold_rows) - 1} rows)"

            summary_rows.append([
                qid,
                "PASS" if match else "FAIL",
                len(gold_rows),
                len(indep_rows),
                sample_display,
            ])

        print(tabulate(
            summary_rows,
            headers=["ID", "Status", "Gold Rows", "Indep Rows", "Sample Result (First Row)"],
            tablefmt="grid",
        ))
        print("=" * 80)

        if all_passed:
            print("SUCCESS: All 10 independent SQL formulations match gold query results exactly!")
        else:
            print("FAILURE: Discrepancy detected between gold and independent formulation.")

        return all_passed

    finally:
        conn.close()


if __name__ == "__main__":
    success = run_spot_check()
    sys.exit(0 if success else 1)
