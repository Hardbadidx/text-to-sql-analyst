"""Pytest fixtures for unit and integration testing."""

import pytest
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)


@pytest.fixture
def test_db_engine():
    """Create a lightweight in-memory SQLite database for fast unit testing."""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()

    Table(
        "patients",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("first_name", String(50), nullable=False),
        Column("last_name", String(50), nullable=False),
        Column("phone", String(30), nullable=False),
        Column("email", String(100), nullable=False),
        Column("address", String(200), nullable=False),
    )

    Table(
        "doctors",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(50), nullable=False),
        Column("specialty", String(50), nullable=False),
    )

    Table(
        "appointments",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("patient_id", Integer, ForeignKey("patients.id"), nullable=False),
        Column("doctor_id", Integer, ForeignKey("doctors.id"), nullable=False),
        Column("scheduled_at", DateTime, nullable=False),
        Column("status", String(20), nullable=False),
    )

    metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            metadata.tables["patients"].insert(),
            [
                {
                    "id": 1,
                    "first_name": "Alice",
                    "last_name": "Smith",
                    "phone": "555-0100",
                    "email": "alice@example.com",
                    "address": "123 Elm St",
                },
                {
                    "id": 2,
                    "first_name": "Bob",
                    "last_name": "Jones",
                    "phone": "555-0101",
                    "email": "bob@example.com",
                    "address": "456 Oak Ave",
                },
            ],
        )
        conn.execute(
            metadata.tables["doctors"].insert(),
            [
                {"id": 1, "name": "Dr. Sarah Adams", "specialty": "Cardiology"},
                {"id": 2, "name": "Dr. Mark Baker", "specialty": "Dermatology"},
            ],
        )

    return engine
