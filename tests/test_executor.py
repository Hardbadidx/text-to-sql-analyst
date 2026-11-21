"""Unit tests for SQLExecutor: read-only database enforcement, query timeouts, and memory limits."""

import time

import pytest

from text2sql.executor import (
    ExecutionResult,
    MemoryLimitExceededError,
    QueryTimeoutError,
    ReadOnlyViolationError,
    SQLExecutor,
)
from text2sql.validator import ValidatedSQL


@pytest.fixture
def executor():
    """Executor connected to clinic.db in read-only mode."""
    return SQLExecutor(database_url="sqlite:///clinic.db", timeout_seconds=5.0)


def test_executor_successful_query(executor):
    """Safe SELECT query returns DataFrame with valid row count and metadata."""
    query = ValidatedSQL("SELECT id, first_name, last_name, specialty FROM doctors ORDER BY id LIMIT 5")
    result = executor.execute(query)

    assert isinstance(result, ExecutionResult)
    assert result.row_count == 5
    assert list(result.df.columns) == ["id", "first_name", "last_name", "specialty"]
    assert result.execution_time_ms > 0
    assert not result.df.empty
    assert "_No rows returned._" not in result.to_markdown_table()
    assert len(result.to_records()) == 5


def test_executor_empty_result(executor):
    """Query yielding 0 rows returns empty DataFrame with correct column names."""
    query = ValidatedSQL("SELECT id, first_name FROM doctors WHERE specialty = 'Astronautics'")
    result = executor.execute(query)

    assert result.row_count == 0
    assert result.df.empty
    assert list(result.df.columns) == ["id", "first_name"]
    assert result.to_markdown_table() == "_No rows returned._"


def test_executor_rejects_unvalidated_raw_string(executor):
    """Executor must strictly reject raw SQL strings without ValidatedSQL wrapper."""
    raw_sql = "SELECT id FROM doctors LIMIT 1"
    with pytest.raises(TypeError) as exc_info:
        executor.execute(raw_sql)  # type: ignore

    assert "requires a ValidatedSQL instance" in str(exc_info.value)
    assert "Unvalidated raw SQL execution is strictly forbidden" in str(exc_info.value)


def test_executor_readonly_blocks_write(executor):
    """Attempting an INSERT or UPDATE must be blocked by SQLite read-only mode."""
    query = ValidatedSQL("UPDATE doctors SET hourly_rate = 500 WHERE id = 1")
    with pytest.raises((ReadOnlyViolationError, Exception)) as exc_info:
        executor.execute(query)
    err = str(exc_info.value).lower()
    assert "readonly" in err or "read-only" in err or "attempt to write a readonly database" in err


def test_executor_readonly_blocks_table_creation(executor):
    """Attempting to CREATE TABLE must fail immediately against read-only database."""
    query = ValidatedSQL("CREATE TABLE malicious_table (id INT)")
    with pytest.raises((ReadOnlyViolationError, Exception)) as exc_info:
        executor.execute(query)
    err = str(exc_info.value).lower()
    assert "readonly" in err or "read-only" in err or "attempt to write a readonly database" in err


def test_executor_memory_length_limit_enforced(executor):
    """Queries attempting to allocate strings/blobs exceeding memory limits must be aborted."""
    # Attempt to allocate 5,000 bytes with limit set to 1,000 bytes
    query = ValidatedSQL("SELECT hex(zeroblob(5000))")
    with pytest.raises(MemoryLimitExceededError) as exc_info:
        executor.execute(query, length_limit_override=1000)

    assert "exceeded maximum length limit" in str(exc_info.value)


def test_executor_cancels_slow_query_with_timeout(executor):
    """Heavy recursive query must trigger the progress handler and be cancelled within tolerance."""
    heavy_query = ValidatedSQL("""
    WITH RECURSIVE infinite_counter(n) AS (
        SELECT 1
        UNION ALL
        SELECT n + 1 FROM infinite_counter
    )
    SELECT COUNT(*) FROM infinite_counter;
    """)

    configured_timeout = 0.10  # 100ms timeout
    start_time = time.perf_counter()

    with pytest.raises(QueryTimeoutError) as exc_info:
        executor.execute(heavy_query, timeout_override=configured_timeout)

    elapsed_seconds = time.perf_counter() - start_time
    print(f"\n[MEASURED TIMEOUT] Configured: {configured_timeout:.2f}s | Elapsed: {elapsed_seconds:.3f}s")

    assert "execution exceeded timeout" in str(exc_info.value)
    # Assert cancellation occurs promptly around timeout threshold (within 350ms window)
    assert 0.08 <= elapsed_seconds <= 0.45


def test_executor_normal_query_not_interrupted(executor):
    """Fast queries finish without being cancelled by the timeout handler."""
    query = ValidatedSQL("SELECT COUNT(*) FROM appointments")
    result = executor.execute(query, timeout_override=3.0)
    assert result.row_count == 1
    assert result.df.iloc[0, 0] == 30000
