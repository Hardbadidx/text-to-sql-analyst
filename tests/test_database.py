"""Database integrity and consistency tests.

Verifies that clinic.db matches the verified cryptographic checksum (data_checksum.txt)
and matches exact expected operational row counts.
"""

import hashlib
import sqlite3
from pathlib import Path

DB_PATH = Path("clinic.db")
CHECKSUM_PATH = Path("data_checksum.txt")

EXPECTED_ROW_COUNTS = {
    "patients": 5000,
    "doctors": 24,
    "appointments": 30000,
    "treatments": 35276,
    "invoices": 23489,
    "payments": 19941,
}


def test_clinic_db_exists_and_sha256_checksum():
    """Assert clinic.db exists and its SHA-256 matches data_checksum.txt."""
    assert DB_PATH.exists(), f"Missing database file at {DB_PATH.resolve()}"
    assert CHECKSUM_PATH.exists(), f"Missing checksum file at {CHECKSUM_PATH.resolve()}"

    expected_sha256 = CHECKSUM_PATH.read_text(encoding="utf-8").strip()
    hasher = hashlib.sha256()
    with open(DB_PATH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_sha256 = hasher.hexdigest()

    assert actual_sha256 == expected_sha256, (
        f"clinic.db SHA256 checksum mismatch!\n"
        f"Expected: {expected_sha256}\n"
        f"Actual:   {actual_sha256}"
    )


def test_clinic_db_row_counts():
    """Assert clinic.db contains exact expected row counts across all 6 operational tables."""
    assert DB_PATH.exists(), f"Missing database file at {DB_PATH.resolve()}"

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        actual_counts = {}
        for table_name in EXPECTED_ROW_COUNTS:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            actual_counts[table_name] = cursor.fetchone()[0]

        assert actual_counts == EXPECTED_ROW_COUNTS, (
            f"Row count mismatch in clinic.db!\n"
            f"Expected: {EXPECTED_ROW_COUNTS}\n"
            f"Actual:   {actual_counts}"
        )
    finally:
        conn.close()
