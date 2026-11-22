"""Unit tests for sqlglot-based AST security validator with adversarial hardening."""

import pytest

from text2sql.validator import SQLValidator, ValidatedSQL


@pytest.fixture
def validator():
    """SQLValidator configured with clinic SaaS schema allowlist and sensitive denylist."""
    allowed_tables = {"patients", "doctors", "appointments", "treatments", "invoices", "payments"}
    allowed_columns = {
        "patients": {"id", "first_name", "last_name", "gender", "date_of_birth", "created_at"},
        "doctors": {"id", "first_name", "last_name", "specialty", "hire_date", "hourly_rate"},
        "appointments": {
            "id",
            "patient_id",
            "doctor_id",
            "scheduled_at",
            "duration_minutes",
            "status",
            "reason",
        },
        "treatments": {"id", "appointment_id", "treatment_code", "description", "cost"},
        "invoices": {"id", "appointment_id", "patient_id", "amount", "status", "issued_at", "due_at"},
        "payments": {"id", "invoice_id", "amount", "payment_date", "payment_method"},
    }
    denylist = ["phone", "email", "address", "ssn", "social_security"]
    return SQLValidator(
        allowed_tables=allowed_tables,
        allowed_columns=allowed_columns,
        denylist_columns=denylist,
        max_query_limit=1000,
    )


# --- 1 to 6: Valid Query Structures & Allowed Functions ---


def test_valid_simple_select(validator):
    sql = "SELECT first_name, last_name FROM patients WHERE gender = 'Female'"
    res = validator.validate(sql)
    assert res.is_valid
    assert isinstance(res.validated_sql, ValidatedSQL)
    assert "LIMIT 1000" in res.sanitized_sql.upper()


def test_valid_join_query(validator):
    sql = (
        "SELECT a.id, d.specialty, a.scheduled_at "
        "FROM appointments a "
        "JOIN doctors d ON a.doctor_id = d.id "
        "WHERE a.status = 'completed'"
    )
    res = validator.validate(sql)
    assert res.is_valid


def test_valid_aggregation_and_group_by(validator):
    sql = (
        "SELECT specialty, COUNT(*) AS doc_count, AVG(hourly_rate) AS avg_rate "
        "FROM doctors GROUP BY specialty HAVING COUNT(*) > 1"
    )
    res = validator.validate(sql)
    assert res.is_valid


def test_valid_cte_query(validator):
    sql = (
        "WITH doctor_appt_counts AS ("
        "  SELECT doctor_id, COUNT(*) AS total_appts "
        "  FROM appointments GROUP BY doctor_id"
        ") "
        "SELECT d.first_name, d.last_name, c.total_appts "
        "FROM doctors d JOIN doctor_appt_counts c ON d.id = c.doctor_id"
    )
    res = validator.validate(sql)
    assert res.is_valid


def test_valid_subquery(validator):
    sql = (
        "SELECT first_name, last_name FROM doctors "
        "WHERE id IN (SELECT doctor_id FROM appointments WHERE status = 'no_show')"
    )
    res = validator.validate(sql)
    assert res.is_valid


def test_valid_union_all(validator):
    sql = (
        "SELECT id, status FROM appointments WHERE status = 'completed' "
        "UNION ALL "
        "SELECT id, status FROM appointments WHERE status = 'cancelled'"
    )
    res = validator.validate(sql)
    assert res.is_valid


def test_valid_allowed_functions(validator):
    """Confirm all standard approved analytical/date functions pass validation."""
    sql = (
        "SELECT "
        "  strftime('%Y-%m', scheduled_at) AS month_str, "
        "  date(scheduled_at) AS date_val, "
        "  ROUND(AVG(duration_minutes), 2) AS avg_duration, "
        "  COALESCE(MAX(duration_minutes), 0) AS max_dur, "
        "  ABS(MIN(duration_minutes)) AS min_dur, "
        "  UPPER(status) AS status_upper, "
        "  LOWER(status) AS status_lower, "
        "  LENGTH(reason) AS reason_len, "
        "  SUBSTR(reason, 1, 10) AS short_reason "
        "FROM appointments "
        "GROUP BY strftime('%Y-%m', scheduled_at), status"
    )
    res = validator.validate(sql)
    assert res.is_valid


# --- 7 to 16: DDL / DML / Command / Injection Attacks ---


def test_reject_insert(validator):
    sql = "INSERT INTO doctors (first_name, last_name) VALUES ('Hacker', 'Doc')"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "Disallowed" in res.error_message


def test_reject_update(validator):
    sql = "UPDATE doctors SET hourly_rate = 999 WHERE id = 1"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "Disallowed" in res.error_message


def test_reject_delete(validator):
    sql = "DELETE FROM patients WHERE id = 1"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "Disallowed" in res.error_message


def test_reject_drop_table(validator):
    sql = "DROP TABLE patients"
    res = validator.validate(sql)
    assert not res.is_valid


def test_reject_alter_table(validator):
    sql = "ALTER TABLE doctors ADD COLUMN secret TEXT"
    res = validator.validate(sql)
    assert not res.is_valid


def test_reject_create_table(validator):
    sql = "CREATE TABLE malicious (id INT)"
    res = validator.validate(sql)
    assert not res.is_valid


def test_reject_attach_database(validator):
    sql = "ATTACH DATABASE 'hack.db' AS hack"
    res = validator.validate(sql)
    assert not res.is_valid


def test_reject_pragma(validator):
    sql = "PRAGMA table_info(patients)"
    res = validator.validate(sql)
    assert not res.is_valid


def test_reject_multi_statement(validator):
    sql = "SELECT id FROM doctors; DROP TABLE appointments;"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "Multiple SQL statements" in res.error_message


def test_reject_comments_hiding_statement(validator):
    sql = "SELECT id FROM doctors -- comment\n; DROP TABLE appointments;"
    res = validator.validate(sql)
    assert not res.is_valid


# --- 17 to 25: Adversarial Hardening Checks ---


def test_adversarial_pragma_table_info_function(validator):
    """PRAGMA table-valued function must be rejected."""
    sql = "SELECT * FROM pragma_table_info('patients')"
    res = validator.validate(sql)
    assert not res.is_valid
    err = res.error_message.lower()
    assert "not allowed" in err or "unauthorized" in err or "disallowed" in err or "forbidden" in err


def test_adversarial_sqlite_master_access(validator):
    """System catalog sqlite_master / sqlite_schema access must be blocked."""
    for table in ["sqlite_master", "sqlite_schema", "sqlite_temp_master"]:
        sql = f"SELECT * FROM {table}"
        res = validator.validate(sql)
        assert not res.is_valid
        assert "forbidden" in res.error_message.lower()


def test_adversarial_table_valued_functions(validator):
    """Table-valued functions like json_each, generate_series must be blocked."""
    for tvf in ["json_each('[]')", "generate_series(1, 10)"]:
        sql = f"SELECT * FROM {tvf}"
        res = validator.validate(sql)
        assert not res.is_valid
        err = res.error_message.lower()
        assert "not allowed" in err or "unauthorized" in err or "disallowed" in err or "forbidden" in err


def test_adversarial_load_extension_call(validator):
    """Dangerous arbitrary code execution functions like load_extension must be blocked."""
    sql = "SELECT load_extension('malicious.dll')"
    res = validator.validate(sql)
    assert not res.is_valid
    assert res.error_message == "Function 'load_extension' is not allowed."


def test_adversarial_disallowed_functions(validator):
    """Functions like zeroblob, randomblob, char must be rejected by the function allowlist."""
    for fn, name in [("zeroblob(100)", "zeroblob"), ("randomblob(100)", "randomblob")]:
        sql = f"SELECT {fn} FROM doctors"
        res = validator.validate(sql)
        assert not res.is_valid
        assert res.error_message == f"Function '{name}' is not allowed."


def test_adversarial_quoted_cased_bracketed_backticked_denylist(validator):
    """Denylisted columns must be caught regardless of quoting, case, brackets, or backticks."""
    variants = [
        'SELECT "PHONE" FROM patients',
        "SELECT [phone] FROM patients",
        "SELECT `email` FROM patients",
        "SELECT PATIENTS.Phone FROM patients",
        "SELECT [patients].[Address] FROM patients",
    ]
    for sql in variants:
        res = validator.validate(sql)
        assert not res.is_valid, f"Failed to reject: {sql}"
        assert "sensitive denylisted column" in res.error_message.lower()


def test_adversarial_join_using_denylisted_column(validator):
    """JOIN ... USING(phone) must be rejected."""
    sql = "SELECT * FROM patients JOIN appointments USING (phone)"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "sensitive denylisted column" in res.error_message.lower()


def test_adversarial_natural_join(validator):
    """NATURAL JOIN must be strictly forbidden."""
    sql = "SELECT * FROM appointments NATURAL JOIN patients"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "NATURAL JOIN is forbidden" in res.error_message


def test_adversarial_cte_shadowing_real_table(validator):
    """CTE alias that shadows a real table name must be blocked."""
    sql = "WITH patients AS (SELECT 1 AS id) SELECT * FROM patients"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "illegally shadows" in res.error_message


# --- 26 to 34: Nested CTEs, Subqueries, Star Expansion & Allowlists ---


def test_reject_unknown_table(validator):
    sql = "SELECT id FROM users"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "unknown or unauthorized table" in res.error_message


def test_reject_unknown_column(validator):
    sql = "SELECT credit_score FROM patients"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "Unknown column" in res.error_message


def test_reject_denylisted_phone_in_select(validator):
    sql = "SELECT id, phone FROM patients"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "sensitive denylisted column 'phone'" in res.error_message


def test_reject_denylisted_email_in_where(validator):
    sql = "SELECT id FROM patients WHERE email LIKE '%@gmail.com'"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "sensitive denylisted column 'email'" in res.error_message


def test_reject_denylisted_address_in_order_by(validator):
    sql = "SELECT id FROM patients ORDER BY address"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "sensitive denylisted column 'address'" in res.error_message


def test_reject_denylisted_column_in_join(validator):
    sql = "SELECT a.id FROM appointments a JOIN patients p ON a.patient_id = p.id WHERE p.ssn = '123'"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "sensitive denylisted column 'ssn'" in res.error_message


def test_expand_star_on_patients_table(validator):
    """SELECT * on patients must expand into only safe columns, omitting phone/email/address."""
    sql = "SELECT * FROM patients"
    res = validator.validate(sql, expand_stars=True)
    assert res.is_valid
    assert res.expanded_stars
    sql_upper = res.sanitized_sql.upper()
    assert "PHONE" not in sql_upper
    assert "EMAIL" not in sql_upper
    assert "ADDRESS" not in sql_upper
    assert "FIRST_NAME" in sql_upper
    assert "DATE_OF_BIRTH" in sql_upper


def test_expand_aliased_star_on_patients(validator):
    """SELECT p.* FROM patients p must expand with alias prefix."""
    sql = "SELECT p.* FROM patients p"
    res = validator.validate(sql, expand_stars=True)
    assert res.is_valid
    assert res.expanded_stars
    assert "p.first_name" in res.sanitized_sql.lower()
    assert "phone" not in res.sanitized_sql.lower()


def test_reject_star_on_patients_when_configured(validator):
    """If reject_stars_on_sensitive is set, wildcard on patients is rejected outright."""
    sql = "SELECT * FROM patients"
    res = validator.validate(sql, reject_stars_on_sensitive=True)
    assert not res.is_valid
    assert "Wildcard 'SELECT *' is blocked" in res.error_message

    # Test qualified rejection as well
    sql_aliased = "SELECT p.* FROM patients p"
    res_aliased = validator.validate(sql_aliased, reject_stars_on_sensitive=True)
    assert not res_aliased.is_valid
    assert "Wildcard 'p.*' is blocked" in res_aliased.error_message


def test_safe_star_on_table_without_denylist(validator):
    """Doctors table has no denylisted columns; SELECT * is permitted."""
    sql = "SELECT * FROM doctors"
    res = validator.validate(sql)
    assert res.is_valid


def test_denylisted_column_in_cte_rejected(validator):
    """Attempting to select phone inside a CTE must be caught and blocked."""
    sql = "WITH leaked AS (SELECT id, phone FROM patients) SELECT * FROM leaked"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "sensitive denylisted column 'phone'" in res.error_message


def test_denylisted_column_in_subquery_rejected(validator):
    """Attempting to select email inside a subquery must be caught and blocked."""
    sql = "SELECT * FROM (SELECT id, email FROM patients) sub"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "sensitive denylisted column 'email'" in res.error_message


def test_star_in_nested_subquery_expanded(validator):
    """SELECT * inside nested subquery on patients must expand safely."""
    sql = "SELECT sub.id, sub.first_name FROM (SELECT * FROM patients) sub"
    res = validator.validate(sql, expand_stars=True)
    assert res.is_valid
    assert "phone" not in res.sanitized_sql.lower()


def test_limit_injected_when_missing(validator):
    sql = "SELECT id, first_name FROM patients"
    res = validator.validate(sql)
    assert res.is_valid
    assert res.clamped_limit
    assert "LIMIT 1000" in res.sanitized_sql.upper()


def test_limit_clamped_when_exceeding_max(validator):
    sql = "SELECT id, first_name FROM patients LIMIT 5000"
    res = validator.validate(sql)
    assert res.is_valid
    assert res.clamped_limit
    assert "LIMIT 1000" in res.sanitized_sql.upper()


def test_safe_small_limit_preserved(validator):
    sql = "SELECT id, first_name FROM patients LIMIT 25"
    res = validator.validate(sql)
    assert res.is_valid
    assert not res.clamped_limit
    assert "LIMIT 25" in res.sanitized_sql.upper()


def test_reject_empty_sql(validator):
    res = validator.validate("   ")
    assert not res.is_valid
    assert "Empty query" in res.error_message


def test_reject_syntax_error(validator):
    res = validator.validate("SELECT FROM WHERE ;;;")
    assert not res.is_valid
    assert "syntax" in res.error_message.lower()


def test_reject_only_comment(validator):
    res = validator.validate("-- this is just a comment")
    assert not res.is_valid


def test_reject_explain_statement(validator):
    res = validator.validate("EXPLAIN SELECT 1")
    assert not res.is_valid
    assert "Disallowed statement type" in res.error_message


def test_non_literal_limit_clamped(validator):
    sql = "SELECT id FROM doctors LIMIT 10 + 5"
    res = validator.validate(sql)
    assert res.is_valid
    assert "LIMIT 1000" in res.sanitized_sql.upper()


def test_reject_vacuum_command(validator):
    """Raw administrative commands like VACUUM must be blocked."""
    res = validator.validate("VACUUM")
    assert not res.is_valid
    assert "Disallowed" in res.error_message or "forbidden" in res.error_message.lower()


def test_expand_star_with_join_on_sensitive_table(validator):
    """SELECT * across a join involving patients must expand safely."""
    sql = "SELECT * FROM appointments a JOIN patients p ON a.patient_id = p.id"
    res = validator.validate(sql, expand_stars=True)
    assert res.is_valid
    assert "phone" not in res.sanitized_sql.lower()
    assert "email" not in res.sanitized_sql.lower()
    assert "address" not in res.sanitized_sql.lower()


def test_dotted_denylisted_identifier(validator):
    """Dotted sensitive column reference like p.phone must be caught."""
    sql = "SELECT p.phone FROM patients p"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "sensitive denylisted column" in res.error_message.lower()


def test_qualified_star_on_safe_table(validator):
    sql = "SELECT d.* FROM doctors d"
    res = validator.validate(sql)
    assert res.is_valid
    assert "d.first_name" in res.sanitized_sql.lower()


def test_star_with_other_columns_on_safe_table(validator):
    sql = "SELECT id, * FROM doctors"
    res = validator.validate(sql)
    assert res.is_valid


def test_validation_result_repr(validator):
    res = validator.validate("SELECT id FROM doctors")
    assert "ValidationResult" in repr(res)
    assert "ValidatedSQL" in repr(res.validated_sql)
    assert str(res.validated_sql) == res.sanitized_sql


# --- Date Rewriting Tests ('now', CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP) ---


def test_rewrite_date_now(validator):
    """date('now') must rewrite to the fixed reference date 2024-12-31."""
    sql = "SELECT date('now')"
    res = validator.validate(sql)
    assert res.is_valid
    assert "2024-12-31" in res.sanitized_sql
    assert "'now'" not in res.sanitized_sql.lower()


def test_rewrite_julianday_now(validator):
    """julianday('now') must rewrite to fixed reference date/timestamp."""
    sql = "SELECT julianday('now')"
    res = validator.validate(sql)
    assert res.is_valid
    assert "2024-12-31" in res.sanitized_sql
    assert "'now'" not in res.sanitized_sql.lower()


def test_rewrite_datetime_now_localtime(validator):
    """datetime('now', 'localtime') must rewrite 'now' to fixed reference date."""
    sql = "SELECT datetime('now', 'localtime')"
    res = validator.validate(sql)
    assert res.is_valid
    assert "2024-12-31" in res.sanitized_sql
    assert "'now'" not in res.sanitized_sql.lower()


def test_rewrite_current_date_and_timestamp(validator):
    """CURRENT_DATE and CURRENT_TIMESTAMP must rewrite to fixed reference strings."""
    sql = "SELECT CURRENT_DATE, CURRENT_TIMESTAMP"
    res = validator.validate(sql)
    assert res.is_valid
    assert "2024-12-31" in res.sanitized_sql
    assert "CURRENT_DATE" not in res.sanitized_sql.upper()


# --- Window Functions & Safe Helpers Tests ---


def test_allowed_window_functions(validator):
    """Confirm window functions pass validation cleanly."""
    sql = (
        "SELECT "
        "  id, doctor_id, duration_minutes, "
        "  ROW_NUMBER() OVER (PARTITION BY doctor_id ORDER BY scheduled_at) AS row_num, "
        "  RANK() OVER (PARTITION BY doctor_id ORDER BY duration_minutes DESC) AS rnk, "
        "  DENSE_RANK() OVER (PARTITION BY doctor_id ORDER BY duration_minutes DESC) AS dense_rnk, "
        "  LAG(duration_minutes, 1) OVER (PARTITION BY doctor_id ORDER BY scheduled_at) AS prev_dur, "
        "  LEAD(duration_minutes, 1) OVER (PARTITION BY doctor_id ORDER BY scheduled_at) AS next_dur, "
        "  NTILE(4) OVER (ORDER BY duration_minutes) AS quartile, "
        "  FIRST_VALUE(duration_minutes) OVER (PARTITION BY doctor_id ORDER BY scheduled_at) AS first_dur, "
        "  LAST_VALUE(duration_minutes) OVER (PARTITION BY doctor_id ORDER BY scheduled_at) AS last_dur, "
        "  IIF(status = 'completed', 1, 0) AS is_done, "
        "  REPLACE(status, 'completed', 'DONE') AS status_rep "
        "FROM appointments"
    )
    res = validator.validate(sql)
    assert res.is_valid


def test_disallowed_functions_still_rejected(validator):
    """Ensure zeroblob, randomblob, and load_extension remain rejected with precise error messages."""
    for fn, name in [
        ("zeroblob(16)", "zeroblob"),
        ("randomblob(16)", "randomblob"),
        ("load_extension('lib')", "load_extension"),
    ]:
        sql = f"SELECT {fn}"
        res = validator.validate(sql)
        assert not res.is_valid
        assert res.error_message == f"Function '{name}' is not allowed."


# --- Regression Tests for Operators, Predicates, and Compound Expressions ---


def test_regression_compound_predicates_and_operators_pass(validator):
    """Ensure operators and predicates are never blocked by the function allowlist."""
    queries_that_must_pass = [
        # The three exact queries from the smoke test
        "SELECT COUNT(*) FROM appointments WHERE status = 'completed'",
        "SELECT d.specialty, ROUND(SUM(i.amount), 2) FROM invoices i JOIN appointments a ON a.id = i.appointment_id JOIN doctors d ON d.id = a.doctor_id GROUP BY d.specialty ORDER BY 2 DESC",
        "SELECT COUNT(*) FROM appointments WHERE status = 'completed' AND scheduled_at >= '2024-01-01' AND scheduled_at <= '2024-12-31 23:59:59'",
        "SELECT COUNT(*) FROM appointments WHERE status = 'completed' AND strftime('%Y', scheduled_at) = '2024'",
        # Nested AND/OR with strftime
        "SELECT * FROM appointments WHERE (status = 'completed' OR status = 'cancelled') AND strftime('%Y', scheduled_at) = '2024'",
        # BETWEEN
        "SELECT * FROM appointments WHERE scheduled_at BETWEEN '2024-01-01' AND '2024-12-31' AND status = 'completed'",
        # IN (subquery)
        "SELECT id, first_name FROM patients WHERE id IN (SELECT patient_id FROM appointments WHERE status = 'completed')",
        # LIKE
        "SELECT id, first_name FROM patients WHERE first_name LIKE 'A%'",
        # IS NULL / IS NOT NULL
        "SELECT * FROM appointments WHERE scheduled_at IS NOT NULL AND status IS NULL",
        # CASE WHEN inside SUM
        "SELECT doctor_id, SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS comp_count FROM appointments GROUP BY doctor_id",
        # Arithmetic in SELECT
        "SELECT id, duration_minutes * 60 + 10 AS total_seconds FROM appointments",
        # String concatenation
        "SELECT first_name || ' ' || last_name AS full_name FROM doctors",
        # HAVING with AND
        "SELECT doctor_id, COUNT(*) FROM appointments GROUP BY doctor_id HAVING COUNT(*) > 5 AND doctor_id IS NOT NULL",
    ]

    for q in queries_that_must_pass:
        res = validator.validate(q)
        assert res.is_valid, f"Expected query to be valid, but got error: {res.error_message}. Query was: {q}"


def test_regression_malicious_functions_inside_expressions_fail(validator):
    """Ensure disallowed functions are blocked even when nested inside compound WHERE or CASE."""
    # zeroblob inside a WHERE with AND
    res_zero = validator.validate(
        "SELECT * FROM appointments WHERE status = 'completed' AND zeroblob(16) IS NOT NULL"
    )
    assert not res_zero.is_valid
    assert res_zero.error_message == "Function 'zeroblob' is not allowed."

    # load_extension inside CASE
    res_ext = validator.validate(
        "SELECT CASE WHEN load_extension('lib') THEN 1 ELSE 0 END FROM appointments"
    )
    assert not res_ext.is_valid
    assert res_ext.error_message == "Function 'load_extension' is not allowed."


# --- Defensive Branch Fail-Closed Tests ---


def test_unresolvable_wildcard_alias_fails_closed(validator):
    """Wildcard with unknown alias like bad.* must fail closed and reject."""
    sql = "SELECT bad.* FROM doctors d"
    res = validator.validate(sql)
    assert not res.is_valid
    assert "unable to resolve wildcard 'bad.*'" in res.error_message.lower()


def test_empty_table_expression_fails_closed(validator):
    """Encountering an empty/unnamed table node must fail closed."""
    # Test validator is_denylisted on dotted notation directly
    assert validator.is_denylisted("patients.phone")
    assert validator.is_denylisted("[patients].[email]")
    assert not validator.is_denylisted("patients.id")


# --- Binary / Predicate Subclass Function Allowlist Tests ---


def test_functions_subclassing_binary_or_predicate_are_blocked(validator):
    """Functions whose AST class subclasses exp.Binary or exp.Predicate must not bypass the allowlist."""
    disallowed = [
        "SELECT json_extract(id, '$.key') FROM appointments",
        "SELECT id FROM appointments WHERE id REGEXP 'abc'",
        "SELECT regexp(id, 'abc') FROM appointments",
        "SELECT zeroblob(64) FROM appointments",
        "SELECT randomblob(64) FROM appointments",
        "SELECT load_extension('evil') FROM appointments",
    ]
    for q in disallowed:
        res = validator.validate(q)
        assert not res.is_valid, f"Expected disallowed function query to fail, but it passed: {q}"
        assert "is not allowed" in res.error_message


def test_allowed_binary_predicate_operators_and_collate_pass(validator):
    """Allowed operators (LIKE, GLOB, BETWEEN, IN, IS NULL), EXISTS, and COLLATE NOCASE must pass."""
    allowed = [
        "SELECT * FROM patients WHERE first_name LIKE '%Jane%'",
        "SELECT * FROM patients WHERE first_name GLOB '*Jane*'",
        "SELECT * FROM appointments WHERE id BETWEEN 1 AND 10",
        "SELECT * FROM appointments WHERE id IN (1, 2, 3)",
        "SELECT * FROM appointments WHERE scheduled_at IS NOT NULL",
        "SELECT id FROM patients WHERE EXISTS (SELECT 1 FROM appointments WHERE appointments.patient_id = patients.id)",
        "SELECT id, first_name FROM patients ORDER BY first_name COLLATE NOCASE",
    ]
    for q in allowed:
        res = validator.validate(q)
        assert res.is_valid, f"Expected valid query to pass, got error: {res.error_message}. Query: {q}"
