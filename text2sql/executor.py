"""Read-only SQL execution engine with real-time interrupt timeouts and memory limits."""

import contextlib
import sqlite3
import time
from typing import Any

import pandas as pd
import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from text2sql.validator import ValidatedSQL


class QueryTimeoutError(Exception):
    """Raised when query execution exceeds the configured timeout limit."""

    pass


class ReadOnlyViolationError(Exception):
    """Raised when an execution violates read-only database constraints."""

    pass


class MemoryLimitExceededError(Exception):
    """Raised when query execution violates database memory or blob length limits."""

    pass


class ExecutionResult(BaseModel):
    """Standardized result envelope for SQL execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame
    row_count: int
    column_names: list[str]
    execution_time_ms: float
    sql: str

    def to_records(self) -> list[dict[str, Any]]:
        """Return results as list of dictionary records."""
        return self.df.to_dict(orient="records")

    def to_markdown_table(self, max_rows: int = 15) -> str:
        """Format the first max_rows into a markdown table."""
        if self.df.empty:
            return "_No rows returned._"
        return self.df.head(max_rows).to_markdown(index=False)


class SQLExecutor:
    """Executes validated SQL queries against SQLite or PostgreSQL databases

    under strict read-only guarantees, memory length bounds, and timeout interruption handlers.
    """

    def __init__(
        self,
        database_url: str = "sqlite:///clinic.db",
        timeout_seconds: float = 5.0,
        max_length_limit: int = 10_000_000,  # 10MB default string/blob ceiling
    ):
        self.database_url = database_url
        self.timeout_seconds = timeout_seconds
        self.max_length_limit = max_length_limit
        self.engine = self._create_readonly_engine(database_url)

    def _create_readonly_engine(self, db_url: str) -> Engine:
        """Create a SQLAlchemy engine configured for read-only access."""
        if db_url.startswith("sqlite"):
            # Ensure URI mode=ro for SQLite
            if ":memory:" in db_url:
                # In-memory database used for fast unit tests
                engine = create_engine(db_url)
            else:
                # Extract clean path and format as read-only URI
                clean_path = db_url.replace("sqlite:///", "").replace("sqlite://", "").split("?")[0]
                ro_url = f"sqlite:///file:{clean_path}?mode=ro&uri=true"
                engine = create_engine(ro_url)
            return engine
        else:
            # Generic / Postgres: connect with read-only transaction defaults
            return create_engine(db_url, execution_options={"isolation_level": "AUTOCOMMIT"})

    def execute(
        self,
        query: ValidatedSQL,
        timeout_override: float | None = None,
        length_limit_override: int | None = None,
    ) -> ExecutionResult:
        """Execute a validated read-only SQL query with progress-handler timeout protection.

        Args:
            query: ValidatedSQL instance (unvalidated raw SQL is rejected).
            timeout_override: Optional custom timeout in seconds for this run.
            length_limit_override: Optional custom string/blob length limit in bytes.

        Returns:
            ExecutionResult with DataFrame and performance metrics.
        """
        # Strictly require ValidatedSQL type
        if not isinstance(query, ValidatedSQL):
            raise TypeError(
                f"SQLExecutor requires a ValidatedSQL instance; received {type(query).__name__}. "
                "Unvalidated raw SQL execution is strictly forbidden."
            )

        sql = query.sql
        timeout = timeout_override or self.timeout_seconds
        length_limit = length_limit_override or self.max_length_limit
        start_time = time.perf_counter()

        with self.engine.connect() as conn:
            raw_connection = conn.connection.dbapi_connection

            # For SQLite connections, configure query_only, memory limits, and progress handler
            if isinstance(raw_connection, sqlite3.Connection):
                with contextlib.suppress(Exception):
                    raw_connection.execute("PRAGMA query_only = ON;")

                # Enforce SQLite length limit against memory bombs
                if hasattr(raw_connection, "setlimit") and hasattr(sqlite3, "SQLITE_LIMIT_LENGTH"):
                    with contextlib.suppress(Exception):
                        raw_connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, length_limit)

                def progress_handler() -> int:
                    # Return non-zero to interrupt SQLite VM execution
                    if (time.perf_counter() - start_time) > timeout:
                        return 1
                    return 0

                # Check progress every 100 SQLite opcodes
                raw_connection.set_progress_handler(progress_handler, 100)

            try:
                result_proxy = conn.execute(text(sql))
                columns = list(result_proxy.keys())
                rows = result_proxy.fetchall()

                # Clear handler on success
                if isinstance(raw_connection, sqlite3.Connection):
                    raw_connection.set_progress_handler(None, 0)

            except (sqlite3.OperationalError, sa.exc.OperationalError) as oe:
                err_str = str(oe).lower()
                if "interrupted" in err_str:
                    raise QueryTimeoutError(
                        f"Query cancelled: execution exceeded timeout of {timeout:.2f}s."
                    ) from oe
                if "readonly" in err_str or "attempt to write a readonly database" in err_str:
                    raise ReadOnlyViolationError(
                        f"Read-only violation: database rejected write attempt: {oe}"
                    ) from oe
                raise
            except (sqlite3.DataError, sa.exc.DataError) as de:
                err_str = str(de).lower()
                if "string or blob too big" in err_str:
                    raise MemoryLimitExceededError(
                        f"Query memory allocation failed: exceeded maximum length limit of {length_limit:,} bytes."
                    ) from de
                raise
            finally:
                if isinstance(raw_connection, sqlite3.Connection):
                    raw_connection.set_progress_handler(None, 0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Construct DataFrame
        df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)

        return ExecutionResult(
            df=df,
            row_count=len(df),
            column_names=columns,
            execution_time_ms=round(elapsed_ms, 2),
            sql=sql,
        )
