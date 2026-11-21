"""Automated Plotly chart selection and generation based on result table heuristics."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def auto_pick_chart_type(df: pd.DataFrame) -> str:
    """Analyze DataFrame structure to recommend the most informative chart type.

    Returns one of: 'bar', 'line', 'pie', 'table-only'.
    """
    if df.empty or len(df) == 0:
        return "table-only"

    # If only 1 row or 1 column, tabular presentation is best
    if len(df) <= 1 or len(df.columns) < 2:
        return "table-only"

    # Identify column types
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [
        c
        for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
        or any(keyword in c.lower() for keyword in ["date", "month", "year", "day", "time"])
    ]
    categorical_cols = [
        c
        for c in df.columns
        if c not in numeric_cols and c not in date_cols and not pd.api.types.is_numeric_dtype(df[c])
    ]

    # Rule 1: Time series / temporal trend -> Line chart
    if date_cols and numeric_cols:
        return "line"

    # Rule 2: Single categorical with few slices (<= 6) representing distribution -> Pie chart
    if len(categorical_cols) == 1 and len(numeric_cols) == 1 and len(df) <= 6:
        col_lower = categorical_cols[0].lower()
        if any(w in col_lower for w in ["status", "gender", "method", "specialty"]):
            return "pie"

    # Rule 3: Categorical grouping with numeric aggregates -> Bar chart
    if (categorical_cols or date_cols) and numeric_cols and len(df) <= 50:
        return "bar"

    # Default fallback
    return "table-only"


def generate_plotly_figure(
    df: pd.DataFrame,
    chart_type: str | None = None,
    title: str = "Query Results",
) -> go.Figure | None:
    """Generate a responsive Plotly figure based on the chosen or auto-detected chart type."""
    if df.empty or len(df.columns) < 2:
        return None

    selected_type = (chart_type or auto_pick_chart_type(df)).lower().strip()
    if selected_type == "table-only":
        return None

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        return None

    x_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
    y_col = numeric_cols[0]

    try:
        if selected_type == "line":
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                markers=True,
                template="plotly_white",
            )
        elif selected_type == "pie":
            fig = px.pie(
                df,
                names=x_col,
                values=y_col,
                title=title,
                template="plotly_white",
                hole=0.3,
            )
        elif selected_type == "bar":
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=title,
                template="plotly_white",
                text_auto=True,
            )
        else:
            return None

        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified" if selected_type != "pie" else "closest",
        )
        return fig
    except Exception:
        # Graceful fallback to table-only on rendering error
        return None
