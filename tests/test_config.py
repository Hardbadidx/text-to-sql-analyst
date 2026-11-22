"""Unit tests for configuration loading, .env parsing, and denylist enforcement."""

from pathlib import Path

from text2sql.config import MANDATORY_DENYLIST, Settings
from text2sql.validator import SQLValidator


def test_settings_loads_from_env_example():
    """Settings must parse .env.example without schema or type decode errors."""
    env_example_path = Path(".env.example")
    assert env_example_path.exists(), ".env.example must exist in project root"

    cfg = Settings(_env_file=env_example_path)
    assert cfg.database_url == "sqlite:///clinic.db"
    assert cfg.llm_provider == "ollama"
    assert cfg.max_query_limit == 1000
    assert cfg.query_timeout_seconds == 5.0
    for col in ["phone", "email", "address", "ssn"]:
        assert col in cfg.denylist_columns


def test_settings_denylist_empty_string_still_blocks_mandatory():
    """An empty string DENYLIST_COLUMNS must still retain all mandatory columns."""
    cfg = Settings(denylist_columns="")
    for col in MANDATORY_DENYLIST:
        assert col in cfg.denylist_columns


def test_settings_denylist_empty_json_list_still_blocks_mandatory():
    """A '[]' JSON string in DENYLIST_COLUMNS must still retain all mandatory columns."""
    cfg = Settings(denylist_columns="[]")
    for col in MANDATORY_DENYLIST:
        assert col in cfg.denylist_columns


def test_settings_denylist_empty_python_list_still_blocks_mandatory():
    """Passing an empty Python list must still retain all mandatory columns."""
    cfg = Settings(denylist_columns=[])
    for col in MANDATORY_DENYLIST:
        assert col in cfg.denylist_columns


def test_settings_denylist_adds_custom_columns_to_mandatory():
    """Additional columns must be added on top of the mandatory minimum denylist."""
    cfg = Settings(denylist_columns="notes,billing_code")
    expected = ["phone", "email", "address", "ssn", "notes", "billing_code"]
    for col in expected:
        assert col in cfg.denylist_columns


def test_validator_with_empty_env_denylist_still_blocks_pii():
    """When DENYLIST_COLUMNS is cleared in env, the validator still blocks phone/email/address."""
    cfg = Settings(denylist_columns=[])
    validator = SQLValidator(
        allowed_tables=["patients"],
        allowed_columns={"patients": ["id", "first_name", "last_name", "phone", "email", "address", "ssn"]},
        denylist_columns=cfg.denylist_columns,
    )

    for sensitive_col in ["phone", "email", "address", "ssn"]:
        query = f"SELECT {sensitive_col} FROM patients"
        result = validator.validate(query)
        assert not result.is_valid, f"Expected {sensitive_col} to be blocked even when denylist is emptied"
        assert sensitive_col in (result.error_message or "").lower()


def test_settings_reads_api_key_from_streamlit_secrets(monkeypatch):
    """When GEMINI_API_KEY is not in env, Settings should read it from Streamlit secrets if present."""
    import sys
    from types import ModuleType

    # Mock streamlit with secrets
    mock_st = ModuleType("streamlit")
    mock_st.secrets = {"GEMINI_API_KEY": "streamlit-secret-test-key-12345"}
    monkeypatch.setitem(sys.modules, "streamlit", mock_st)

    cfg = Settings(gemini_api_key=None)
    assert cfg.gemini_api_key == "streamlit-secret-test-key-12345"

