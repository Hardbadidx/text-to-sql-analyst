"""Unit tests for automated Plotly chart selection and generation."""

import pandas as pd

from text2sql.charts import auto_pick_chart_type, generate_plotly_figure


def test_chart_selection_line_for_temporal_data():
    df = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02", "2024-03"],
            "total_revenue": [12000.0, 15000.0, 14000.0],
        }
    )
    chart_type = auto_pick_chart_type(df)
    assert chart_type == "line"

    fig = generate_plotly_figure(df, chart_type=chart_type)
    assert fig is not None


def test_chart_selection_pie_for_small_categorical_distribution():
    df = pd.DataFrame(
        {
            "status": ["completed", "cancelled", "no_show"],
            "count": [23000, 3500, 3000],
        }
    )
    chart_type = auto_pick_chart_type(df)
    assert chart_type == "pie"

    fig = generate_plotly_figure(df, chart_type=chart_type)
    assert fig is not None


def test_chart_selection_bar_for_general_categories():
    df = pd.DataFrame(
        {
            "doctor": [f"Dr. {i}" for i in range(10)],
            "completed_appointments": [120 + i for i in range(10)],
        }
    )
    chart_type = auto_pick_chart_type(df)
    assert chart_type == "bar"

    fig = generate_plotly_figure(df, chart_type=chart_type)
    assert fig is not None


def test_chart_selection_table_only_for_single_row():
    df = pd.DataFrame({"total_patients": [5000]})
    chart_type = auto_pick_chart_type(df)
    assert chart_type == "table-only"

    fig = generate_plotly_figure(df, chart_type=chart_type)
    assert fig is None


def test_chart_selection_table_only_for_empty_df():
    df = pd.DataFrame()
    chart_type = auto_pick_chart_type(df)
    assert chart_type == "table-only"
    assert generate_plotly_figure(df) is None
