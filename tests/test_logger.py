"""Unit tests for AuditLogger safety, lazy filesystem operations, and graceful fallback.

Verifies:
1. AuditLogger.__init__ produces NO filesystem side effects (no mkdir, no file creation).
2. Importing text2sql.agent produces NO filesystem side effects.
3. Directory creation occurs lazily on the first successful write.
4. If directory creation or file writing fails with PermissionError/OSError,
   logging is disabled gracefully without raising exceptions, emitting a single warning.
"""

import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from text2sql.logger import AuditLogger, QueryAuditLog


def test_audit_logger_init_has_no_filesystem_side_effects(tmp_path):
    """Instantiating AuditLogger must not create any directory or file."""
    non_existent_dir = tmp_path / "lazy_dir_test"
    target_log_file = non_existent_dir / "audit.jsonl"

    assert not non_existent_dir.exists()

    logger = AuditLogger(log_path=target_log_file)
    assert logger.log_path == target_log_file
    # Directory and file must still NOT exist after instantiation
    assert not non_existent_dir.exists()
    assert not target_log_file.exists()


def test_audit_logger_lazy_directory_creation_on_first_write(tmp_path):
    """AuditLogger creates parent directories only when the first record is logged."""
    non_existent_dir = tmp_path / "lazy_created_dir"
    target_log_file = non_existent_dir / "audit.jsonl"

    logger = AuditLogger(log_path=target_log_file)
    assert not non_existent_dir.exists()

    entry = logger.log(
        question="How many patients?",
        final_sql="SELECT COUNT(*) FROM patients",
        attempts=1,
        status="ok",
        latency_ms=120.5,
        row_count=1,
    )

    assert isinstance(entry, QueryAuditLog)
    assert non_existent_dir.exists()
    assert target_log_file.exists()
    content = target_log_file.read_text(encoding="utf-8")
    assert "How many patients?" in content


def test_audit_logger_permission_error_on_mkdir_fallback(tmp_path):
    """If parent.mkdir raises PermissionError, AuditLogger must not raise and must disable logging."""
    target_log_file = tmp_path / "unwritable_dir" / "audit.jsonl"
    logger = AuditLogger(log_path=target_log_file)

    with patch.object(Path, "mkdir", side_effect=PermissionError(13, "Permission denied")):
        with pytest.warns(RuntimeWarning, match="unable to create log directory"):
            entry1 = logger.log(
                question="Test question 1",
                final_sql="SELECT 1",
                attempts=1,
                status="ok",
                latency_ms=50.0,
            )

        assert isinstance(entry1, QueryAuditLog)
        assert entry1.question == "Test question 1"

        # Subsequent call should NOT re-raise, should NOT emit duplicate warning, and returns entry
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            entry2 = logger.log(
                question="Test question 2",
                final_sql="SELECT 2",
                attempts=1,
                status="ok",
                latency_ms=45.0,
            )
            assert isinstance(entry2, QueryAuditLog)
            # No new warning emitted because logger was cleanly disabled
            assert len(record) == 0


def test_audit_logger_permission_error_on_file_write_fallback(tmp_path):
    """If opening the log file raises PermissionError, AuditLogger must not raise and must disable logging."""
    log_dir = tmp_path / "write_fail_dir"
    log_dir.mkdir(parents=True, exist_ok=True)
    target_log_file = log_dir / "audit.jsonl"
    logger = AuditLogger(log_path=target_log_file)

    with patch("builtins.open", side_effect=PermissionError(13, "Permission denied")):
        with pytest.warns(RuntimeWarning, match="unable to write to log file"):
            entry = logger.log(
                question="Test write failure",
                final_sql="SELECT 1",
                attempts=1,
                status="ok",
                latency_ms=10.0,
            )
        assert isinstance(entry, QueryAuditLog)


def test_import_text2sql_agent_creates_no_filesystem_side_effects(tmp_path, monkeypatch):
    """Importing text2sql.agent must not trigger any directory creation."""
    # Run in a clean isolated test context
    isolated_dir = tmp_path / "isolated_import_dir"
    assert not isolated_dir.exists()

    # Verify that even with settings pointing to a non-existent path, importing agent does not mkdir
    monkeypatch.setenv("LOG_FILE", str(isolated_dir / "audit.jsonl"))

    # Reload text2sql.logger and text2sql.agent to test import-time behavior
    if "text2sql.agent" in sys.modules:
        import importlib

        import text2sql.logger
        importlib.reload(text2sql.logger)
        import text2sql.agent
        importlib.reload(text2sql.agent)
    else:
        import text2sql.agent  # noqa: F401

    assert not isolated_dir.exists(), "Importing text2sql.agent created a directory as side effect!"
