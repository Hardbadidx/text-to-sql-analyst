"""Unit tests for database schema introspection and denylist enforcement."""

from text2sql.schema import introspect_schema


def test_schema_introspect_excludes_denylisted_columns(test_db_engine):
    """Denylisted columns (phone, email, address) must never appear in introspected schema."""
    denylist = ["phone", "email", "address"]
    schema = introspect_schema(test_db_engine, denylist=denylist)

    assert "patients" in schema.tables
    patient_cols = [c.name for c in schema.tables["patients"].columns]

    assert "id" in patient_cols
    assert "first_name" in patient_cols
    assert "last_name" in patient_cols
    assert "phone" not in patient_cols
    assert "email" not in patient_cols
    assert "address" not in patient_cols


def test_schema_sample_values_exclude_denylisted_columns(test_db_engine):
    """Sample values must never be queried or stored for denylisted columns."""
    denylist = ["phone", "email", "address"]
    schema = introspect_schema(test_db_engine, denylist=denylist)

    patients_table = schema.tables["patients"]
    for col in patients_table.columns:
        assert col.name not in ["phone", "email", "address"]
        if col.name == "first_name":
            assert len(col.sample_values) > 0
            assert "Alice" in col.sample_values or "Bob" in col.sample_values


def test_schema_prompt_context_omits_sensitive_data(test_db_engine):
    """Generated prompt context for LLM must contain zero trace of denylisted columns."""
    denylist = ["phone", "email", "address"]
    schema = introspect_schema(test_db_engine, denylist=denylist)
    prompt_str = schema.to_prompt_context()

    assert "phone" not in prompt_str.lower()
    assert "email" not in prompt_str.lower()
    assert "address" not in prompt_str.lower()
    assert "555-0100" not in prompt_str
    assert "alice@example.com" not in prompt_str
    assert "123 elm st" not in prompt_str.lower()
    assert "first_name" in prompt_str
    assert "Cardiology" in prompt_str or "specialty" in prompt_str
    assert "CURRENT_DATE" in prompt_str


def test_allowed_columns_mapping(test_db_engine):
    """Allowed columns mapping should only list safe, non-denylisted columns."""
    schema = introspect_schema(test_db_engine, denylist=["phone", "email", "address"])
    allowed = schema.get_allowed_columns()

    assert "patients" in allowed
    assert allowed["patients"] == {"id", "first_name", "last_name"}
    assert "phone" not in allowed["patients"]
    assert "email" not in allowed["patients"]
    assert "address" not in allowed["patients"]
