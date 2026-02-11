"""Trend visualization utilities for saved reports."""

import json
import logging
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from orm.models import ReportExecution

logger = logging.getLogger(__name__)


def build_trend_chart(
    executions: list[ReportExecution],
    column: str,
    metric: str = "sum",
) -> go.Figure | None:
    """Build a Plotly line chart showing metric values over time.

    Args:
        executions: List of ReportExecution objects (should be ordered by created_at)
        column: The column name to chart
        metric: The metric to display (sum, avg, min, max, count)

    Returns:
        Plotly Figure or None if no data
    """
    if not executions:
        return None

    timestamps = []
    values = []

    for exec in executions:
        if not exec.success or not exec.numeric_summary:
            continue

        try:
            summary = json.loads(exec.numeric_summary)
        except (json.JSONDecodeError, TypeError):
            continue

        if column not in summary or metric not in summary[column]:
            continue

        timestamps.append(exec.created_at)
        values.append(summary[column][metric])

    if not timestamps:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=values,
            mode="lines+markers",
            name=f"{column} ({metric})",
            line=dict(width=2),
            marker=dict(size=8),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>" + f"{metric}: %{{y:.2f}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{column} - {metric.capitalize()} Over Time",
        xaxis_title="Execution Date",
        yaxis_title=metric.capitalize(),
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    return fig


def build_multi_metric_chart(
    executions: list[ReportExecution],
    column: str,
) -> go.Figure | None:
    """Build a chart showing multiple metrics for a column over time.

    Args:
        executions: List of ReportExecution objects
        column: The column name to chart

    Returns:
        Plotly Figure with multiple traces or None if no data
    """
    if not executions:
        return None

    # Collect data for each metric
    data = {"timestamp": [], "sum": [], "avg": [], "min": [], "max": []}

    for exec in executions:
        if not exec.success or not exec.numeric_summary:
            continue

        try:
            summary = json.loads(exec.numeric_summary)
        except (json.JSONDecodeError, TypeError):
            continue

        if column not in summary:
            continue

        data["timestamp"].append(exec.created_at)
        for metric in ["sum", "avg", "min", "max"]:
            data[metric].append(summary[column].get(metric))

    if not data["timestamp"]:
        return None

    fig = go.Figure()

    colors = {"sum": "#1f77b4", "avg": "#ff7f0e", "min": "#2ca02c", "max": "#d62728"}

    for metric in ["sum", "avg", "min", "max"]:
        if any(v is not None for v in data[metric]):
            fig.add_trace(
                go.Scatter(
                    x=data["timestamp"],
                    y=data[metric],
                    mode="lines+markers",
                    name=metric.capitalize(),
                    line=dict(color=colors[metric], width=2),
                    marker=dict(size=6),
                )
            )

    fig.update_layout(
        title=f"{column} - All Metrics Over Time",
        xaxis_title="Execution Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def build_row_count_chart(executions: list[ReportExecution]) -> go.Figure | None:
    """Build a simple chart showing row count over time.

    Args:
        executions: List of ReportExecution objects

    Returns:
        Plotly Figure or None if no data
    """
    if not executions:
        return None

    timestamps = []
    row_counts = []

    for exec in executions:
        if not exec.success:
            continue
        timestamps.append(exec.created_at)
        row_counts.append(exec.row_count or 0)

    if not timestamps:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=row_counts,
            name="Row Count",
            marker_color="#1f77b4",
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Rows: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Row Count Over Time",
        xaxis_title="Execution Date",
        yaxis_title="Row Count",
        template="plotly_white",
        height=300,
    )

    return fig


def build_comparison_table(
    exec1: ReportExecution,
    exec2: ReportExecution,
) -> pd.DataFrame | None:
    """Build a comparison DataFrame between two executions.

    Args:
        exec1: First (older) execution
        exec2: Second (newer) execution

    Returns:
        DataFrame with comparison data or None if invalid
    """
    if not exec1 or not exec2:
        return None

    if not exec1.numeric_summary or not exec2.numeric_summary:
        return None

    try:
        summary1 = json.loads(exec1.numeric_summary)
        summary2 = json.loads(exec2.numeric_summary)
    except (json.JSONDecodeError, TypeError):
        return None

    # Get all columns from both
    all_cols = set(summary1.keys()) | set(summary2.keys())

    rows = []
    for col in sorted(all_cols):
        for metric in ["count", "sum", "avg", "min", "max"]:
            val1 = summary1.get(col, {}).get(metric)
            val2 = summary2.get(col, {}).get(metric)

            if val1 is None and val2 is None:
                continue

            # Calculate delta
            delta = None
            delta_str = ""
            if val1 is not None and val2 is not None:
                delta = val2 - val1
                if val1 != 0:
                    pct = (delta / val1) * 100
                    if pct > 0:
                        delta_str = f"↑ {pct:.1f}%"
                    elif pct < 0:
                        delta_str = f"↓ {abs(pct):.1f}%"
                    else:
                        delta_str = "→ 0%"
                elif delta != 0:
                    delta_str = "↑ ∞" if delta > 0 else "↓ ∞"
                else:
                    delta_str = "→ 0%"

            rows.append(
                {
                    "Column": col,
                    "Metric": metric.capitalize(),
                    "Previous": f"{val1:.2f}" if val1 is not None else "N/A",
                    "Current": f"{val2:.2f}" if val2 is not None else "N/A",
                    "Change": delta_str,
                }
            )

    return pd.DataFrame(rows) if rows else None


def build_sparkline_data(executions: list[ReportExecution], column: str, metric: str = "sum") -> list[float]:
    """Extract data points for a mini sparkline chart.

    Args:
        executions: List of ReportExecution objects (ordered by created_at)
        column: Column to extract
        metric: Metric to use

    Returns:
        List of float values for sparkline
    """
    values = []
    for exec in executions:
        if not exec.success or not exec.numeric_summary:
            continue
        try:
            summary = json.loads(exec.numeric_summary)
            if column in summary and metric in summary[column]:
                values.append(summary[column][metric])
        except (json.JSONDecodeError, TypeError):
            continue
    return values


def build_execution_history_df(executions: list[ReportExecution]) -> pd.DataFrame:
    """Build a DataFrame summarizing execution history.

    Args:
        executions: List of ReportExecution objects

    Returns:
        DataFrame with execution history summary
    """
    if not executions:
        return pd.DataFrame()

    rows = []
    for exec in executions:
        rows.append(
            {
                "Date": exec.created_at.strftime("%Y-%m-%d %H:%M") if exec.created_at else "N/A",
                "Rows": exec.row_count if exec.row_count is not None else 0,
                "Time (s)": f"{float(exec.elapsed_time):.2f}" if exec.elapsed_time else "N/A",
                "Status": "✓ Success" if exec.success else f"✗ {exec.error_message[:30]}..." if exec.error_message else "✗ Failed",
                "Triggered By": exec.triggered_by or "manual",
            }
        )

    return pd.DataFrame(rows)


def format_time_ago(dt: datetime) -> str:
    """Format a datetime as a human-readable 'time ago' string.

    Args:
        dt: The datetime to format

    Returns:
        String like "2 hours ago", "3 days ago", etc.
    """
    if not dt:
        return "Never"

    now = datetime.now()
    if dt.tzinfo:
        # If dt is timezone-aware, make now timezone-aware too
        from datetime import timezone

        now = datetime.now(timezone.utc)

    diff = now - dt

    seconds = diff.total_seconds()
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
