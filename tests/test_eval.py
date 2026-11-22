"""Unit tests for the Stage 4 evaluation harness, comparator, and caching."""

import pandas as pd
import pytest

from eval.run_eval import (
    EvalCache,
    Evaluator,
    RateLimiter,
    compare_dataframes,
    compare_dataframes_exact,
    compare_dataframes_lenient,
    compare_dataframes_strict,
    get_git_commit_hash,
)
from text2sql.agent import Text2SQLAgent
from text2sql.llm.base import ProviderUnavailableError
from text2sql.llm.mock import MockLLMProvider


def test_comparator_extra_columns():
    """Predicted DataFrame with extra columns must fail exact comparator but pass lenient comparator."""
    df_gold = pd.DataFrame({"id": [1, 2], "val": [10.5, 20.3]})
    df_pred_extra = pd.DataFrame({"id": [1, 2], "val": [10.5, 20.3], "extra_metric": ["x", "y"]})

    assert not compare_dataframes_exact(df_pred_extra, df_gold)
    assert compare_dataframes_lenient(df_pred_extra, df_gold)


def test_comparator_column_order():
    """Predicted DataFrame with permuted column order must pass both exact and lenient comparators."""
    df_gold = pd.DataFrame({"id": [1, 2], "val": [10.5, 20.3]})
    df_pred_reordered = pd.DataFrame({"val": [10.5, 20.3], "id": [1, 2]})

    assert compare_dataframes_exact(df_pred_reordered, df_gold)
    assert compare_dataframes_lenient(df_pred_reordered, df_gold)


def test_comparator_nan_rows():
    """NULL/None/NaN in predicted and gold must be treated as equal in both strict and lenient comparators."""
    import numpy as np

    df_gold = pd.DataFrame({"id": [1, 2], "val": [np.nan, 20.0]})
    df_pred = pd.DataFrame({"id": [1, 2], "val": [None, 20.0]})

    assert compare_dataframes_strict(df_pred, df_gold)
    assert compare_dataframes_lenient(df_pred, df_gold)


def test_comparator_float_rounding():
    """Floats differing beyond 2 decimal places must match in both strict and lenient comparators."""
    df_gold = pd.DataFrame({"val": [12.35]})
    df_pred = pd.DataFrame({"val": [12.3456]})

    assert compare_dataframes_strict(df_pred, df_gold)
    assert compare_dataframes_lenient(df_pred, df_gold)


def test_comparator_ties():
    """Tied identical rows must match with duplicate counts preserved in both strict and lenient comparators."""
    df_gold = pd.DataFrame({"specialty": ["Cardiology", "Cardiology"], "cnt": [5, 5]})
    df_pred = pd.DataFrame({"specialty": ["Cardiology", "Cardiology"], "cnt": [5, 5]})

    assert compare_dataframes_strict(df_pred, df_gold)
    assert compare_dataframes_lenient(df_pred, df_gold)


def test_compare_dataframes_order_insensitive_and_aliases():
    """Comparator must match identical data despite column renames or row order differences."""
    df1 = pd.DataFrame({"total": [5000.001], "cat": ["Cardiology "]})
    df2 = pd.DataFrame({"count": [5000.0], "specialty": ["cardiology"]})
    assert compare_dataframes(df1, df2)

    df_a = pd.DataFrame({"id": [1, 2], "val": [10.5, 20.3]})
    df_b = pd.DataFrame({"num": [2, 1], "amount": [20.3, 10.5]})
    assert compare_dataframes(df_a, df_b)

    df_diff = pd.DataFrame({"id": [1, 2], "val": [10.5, 99.9]})
    assert not compare_dataframes(df_a, df_diff)


def test_eval_cache_disk_persistence(tmp_path):
    """EvalCache must persist and retrieve JSON entries keyed by model and question."""
    cache = EvalCache(cache_dir=str(tmp_path))
    model = "test-model"
    q = "How many patients?"

    assert cache.get(model, q) is None

    payload = {"status": "ok", "sql": "SELECT COUNT(*) FROM patients", "is_correct": True}
    cache.set(model, q, payload)

    retrieved = cache.get(model, q)
    assert retrieved is not None
    assert retrieved["sql"] == payload["sql"]
    assert retrieved["is_correct"] is True


def test_rate_limiter_min_interval():
    """RateLimiter configured for 12 req/min must enforce 5s interval."""
    limiter = RateLimiter(max_per_minute=12)
    assert limiter.min_interval == pytest.approx(5.0)


def test_git_commit_hash_retrieval():
    """git commit hash should be a non-empty string."""
    commit_hash = get_git_commit_hash()
    assert isinstance(commit_hash, str)
    assert len(commit_hash) > 0


def test_evaluator_with_mock_provider(tmp_path):
    """Evaluator must run on mock provider, compute metrics, and write latest.json."""
    mock_provider = MockLLMProvider(
        default_response={
            "sql": "SELECT COUNT(*) AS total_patients FROM patients",
            "reasoning_summary": "Selecting count of patients.",
            "is_refusal": False,
        }
    )
    test_agent = Text2SQLAgent(llm_provider=mock_provider)

    evaluator = Evaluator(
        questions_path="eval/questions.yaml",
        safety_path="eval/safety_cases.yaml",
        results_dir=str(tmp_path / "results"),
        cache_dir=str(tmp_path / "cache"),
        custom_agent=test_agent,
    )

    # Evaluate the first 3 questions using the mock provider
    report = evaluator.run(max_questions=3)

    assert "metadata" in report
    assert "metrics" in report
    assert report["metadata"]["total_questions"] == 3
    assert (tmp_path / "results" / "latest.json").exists()

    metrics = report["metrics"]
    assert metrics["evaluated_questions"] == 3
    assert metrics["safety_block_rate_pct"] == 100.0


def test_evaluator_handles_daily_quota_gracefully(tmp_path):
    """When a 429 PerDay error occurs, unrun questions must be marked 'not run' and exit cleanly."""
    mock_provider = MockLLMProvider(
        responses=[
            ProviderUnavailableError("Gemini daily API quota reached (quotaId contains PerDay): 429 RESOURCE_EXHAUSTED")
        ]
    )
    test_agent = Text2SQLAgent(llm_provider=mock_provider)

    evaluator = Evaluator(
        questions_path="eval/questions.yaml",
        safety_path="eval/safety_cases.yaml",
        results_dir=str(tmp_path / "results"),
        cache_dir=str(tmp_path / "cache"),
        custom_agent=test_agent,
    )

    # Run on 3 questions; question 1 hits 429 PerDay, questions 2 and 3 must be marked 'not run'
    report = evaluator.run(max_questions=3)

    assert report["metadata"]["quota_exhausted"] is True
    results = report["results"]
    assert results[0]["status"] == "provider_unavailable"
    assert results[1]["status"] == "not run"
    assert results[2]["status"] == "not run"
    assert report["metrics"]["not_run_count"] == 2
