"""Service layer for Saved Reports with trend tracking functionality."""

import json
import logging
from datetime import datetime
from difflib import SequenceMatcher

import pandas as pd

from orm.models import ReportExecution, SavedReport, SessionLocal

logger = logging.getLogger(__name__)

# Similarity threshold for matching questions to saved reports
SIMILARITY_THRESHOLD = 0.90


def _normalize_question(text: str) -> str:
    """Normalize a question by removing common prefixes and filler words.

    This improves matching for semantically identical questions like:
    - "what were the top diagnosis" vs "top diagnosis"
    - "show me patients by age" vs "patients by age"
    """
    if not text:
        return ""

    normalized = text.lower().strip()

    # Common question prefixes to strip (order matters - longer first)
    prefixes = [
        "can you show me ",
        "can you tell me ",
        "can you give me ",
        "can you list ",
        "please show me ",
        "please tell me ",
        "please give me ",
        "please list ",
        "what were the ",
        "what are the ",
        "what is the ",
        "what was the ",
        "show me the ",
        "give me the ",
        "tell me the ",
        "list the ",
        "show me ",
        "give me ",
        "tell me ",
        "list ",
        "what ",
        "how many ",
        "how much ",
    ]

    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break  # Only strip one prefix

    return normalized.strip()


def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two strings (0-1).

    Normalizes questions to remove common prefixes before comparison.
    """
    if not text1 or not text2:
        return 0.0

    # Normalize both texts
    t1 = _normalize_question(text1)
    t2 = _normalize_question(text2)

    return SequenceMatcher(None, t1, t2).ratio()


def extract_numeric_summary(df: pd.DataFrame) -> dict:
    """Extract summary statistics for all numeric columns in a DataFrame.

    Returns dict like: {column_name: {sum, avg, min, max, count}}
    """
    if df is None or df.empty:
        return {}

    summary = {}
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            summary[col] = {
                "sum": float(col_data.sum()),
                "avg": float(col_data.mean()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "count": int(len(col_data)),
            }

    return summary


# ==============================================================================
# Core CRUD Operations
# ==============================================================================


def save_report_from_chat(
    user_id: int,
    name: str,
    sql_query: str,
    original_question: str,
    source_group_id: str = None,
    description: str = None,
) -> SavedReport:
    """Create a new saved report from chat context.

    Args:
        user_id: The user who is saving the report
        name: User-provided name for the report
        sql_query: The SQL query to save (frozen, never regenerated)
        original_question: The natural language question that generated the SQL
        source_group_id: Optional group_id from the chat message
        description: Optional description

    Returns:
        The created SavedReport object
    """
    session = SessionLocal()
    try:
        report = SavedReport(
            user_id=user_id,
            name=name,
            sql_query=sql_query,
            original_question=original_question,
            source_group_id=source_group_id,
            description=description,
        )
        session.add(report)
        session.commit()
        session.refresh(report)
        logger.info(f"Saved report '{name}' (id={report.id}) for user {user_id}")
        return report
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save report: {e}")
        raise
    finally:
        session.close()


def get_user_reports(user_id: int, include_archived: bool = False) -> list[SavedReport]:
    """Get all saved reports for a user.

    Args:
        user_id: The user whose reports to retrieve
        include_archived: Whether to include archived reports

    Returns:
        List of SavedReport objects ordered by most recent first
    """
    session = SessionLocal()
    try:
        query = session.query(SavedReport).filter(SavedReport.user_id == user_id)
        if not include_archived:
            query = query.filter(SavedReport.is_archived == False)  # noqa: E712
        return query.order_by(SavedReport.updated_at.desc()).all()
    finally:
        session.close()


def get_report_by_id(report_id: int) -> SavedReport | None:
    """Get a single report by ID."""
    session = SessionLocal()
    try:
        return session.query(SavedReport).filter(SavedReport.id == report_id).first()
    finally:
        session.close()


def get_report_by_name(user_id: int, name: str) -> SavedReport | None:
    """Get a report by name for a specific user (case-insensitive)."""
    session = SessionLocal()
    try:
        # SQLite uses LIKE for case-insensitive, PostgreSQL uses ILIKE
        return (
            session.query(SavedReport)
            .filter(SavedReport.user_id == user_id, SavedReport.name.ilike(name), SavedReport.is_archived == False)  # noqa: E712
            .first()
        )
    finally:
        session.close()


def update_report(report_id: int, user_id: int, **kwargs) -> SavedReport | None:
    """Update a saved report.

    Args:
        report_id: The report to update
        user_id: The user making the update (must own the report)
        **kwargs: Fields to update (name, description, tags)

    Returns:
        Updated SavedReport or None if not found/not authorized
    """
    session = SessionLocal()
    try:
        report = (
            session.query(SavedReport).filter(SavedReport.id == report_id, SavedReport.user_id == user_id).first()
        )
        if not report:
            return None

        for key, value in kwargs.items():
            if hasattr(report, key) and key not in ("id", "user_id", "created_at"):
                setattr(report, key, value)

        session.commit()
        session.refresh(report)
        return report
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to update report {report_id}: {e}")
        raise
    finally:
        session.close()


def archive_report(report_id: int, user_id: int) -> bool:
    """Archive a saved report (soft delete).

    Args:
        report_id: The report to archive
        user_id: The user making the request (must own the report)

    Returns:
        True if archived, False if not found/not authorized
    """
    session = SessionLocal()
    try:
        report = (
            session.query(SavedReport).filter(SavedReport.id == report_id, SavedReport.user_id == user_id).first()
        )
        if not report:
            return False

        report.is_archived = True
        session.commit()
        logger.info(f"Archived report {report_id}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to archive report {report_id}: {e}")
        return False
    finally:
        session.close()


# ==============================================================================
# Report Execution
# ==============================================================================


def execute_report(
    report_id: int,
    user_id: int,
    triggered_by: str = "manual",
    group_id: str = None,
) -> tuple[ReportExecution, pd.DataFrame | None]:
    """Execute a saved report and record the execution.

    Args:
        report_id: The report to execute
        user_id: The user executing the report
        triggered_by: How this was triggered ("manual", "scheduled", "api", "auto")
        group_id: Optional chat group_id to link this execution

    Returns:
        Tuple of (ReportExecution, DataFrame or None if failed)
    """
    import time

    from utils.vanna_calls import VannaService

    session = SessionLocal()
    try:
        report = session.query(SavedReport).filter(SavedReport.id == report_id).first()
        if not report:
            raise ValueError(f"Report {report_id} not found")

        # Execute the SQL
        vn = VannaService.from_streamlit_session()
        start_time = time.time()

        try:
            result = vn.run_sql(sql=report.sql_query)
            elapsed_time = time.time() - start_time

            if isinstance(result, tuple):
                df, _ = result
            else:
                df = result

            if isinstance(df, pd.DataFrame):
                # Successful execution
                execution = ReportExecution(
                    report_id=report_id,
                    triggered_by=triggered_by,
                    executed_by_user_id=user_id,
                    dataframe_json=df.to_json(date_format="iso"),
                    row_count=len(df),
                    column_names=json.dumps(list(df.columns)),
                    elapsed_time=elapsed_time,
                    success=True,
                    numeric_summary=json.dumps(extract_numeric_summary(df)),
                    group_id=group_id,
                )
            else:
                # Execution returned non-DataFrame
                execution = ReportExecution(
                    report_id=report_id,
                    triggered_by=triggered_by,
                    executed_by_user_id=user_id,
                    elapsed_time=elapsed_time,
                    success=False,
                    error_message="Query did not return a DataFrame",
                    group_id=group_id,
                )
                df = None

        except Exception as e:
            elapsed_time = time.time() - start_time
            execution = ReportExecution(
                report_id=report_id,
                triggered_by=triggered_by,
                executed_by_user_id=user_id,
                elapsed_time=elapsed_time,
                success=False,
                error_message=str(e)[:2000],
                group_id=group_id,
            )
            df = None
            logger.error(f"Report execution failed for report {report_id}: {e}")

        session.add(execution)
        session.commit()
        session.refresh(execution)

        logger.info(
            f"Recorded execution for report {report_id}: success={execution.success}, rows={execution.row_count}"
        )
        return execution, df

    finally:
        session.close()


def record_report_execution(
    report: SavedReport,
    df: pd.DataFrame,
    user_id: int,
    group_id: str,
    elapsed_time: float = 0,
) -> ReportExecution:
    """Record an execution when a question matches a saved report.

    Called automatically from normal_message_flow when match detected.

    Args:
        report: The matched SavedReport
        df: The DataFrame result from executing the query
        user_id: The user who ran the query
        group_id: The chat group_id
        elapsed_time: Query execution time

    Returns:
        The created ReportExecution
    """
    session = SessionLocal()
    try:
        execution = ReportExecution(
            report_id=report.id,
            triggered_by="auto",
            executed_by_user_id=user_id,
            dataframe_json=df.to_json(date_format="iso") if df is not None else None,
            row_count=len(df) if df is not None else 0,
            column_names=json.dumps(list(df.columns)) if df is not None else None,
            elapsed_time=elapsed_time,
            success=df is not None,
            numeric_summary=json.dumps(extract_numeric_summary(df)) if df is not None else None,
            group_id=group_id,
        )
        session.add(execution)
        session.commit()
        session.refresh(execution)

        logger.info(f"Auto-recorded execution for report '{report.name}' (id={report.id})")
        return execution
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to record report execution: {e}")
        raise
    finally:
        session.close()


def get_report_executions(report_id: int, limit: int = 50) -> list[ReportExecution]:
    """Get execution history for a report.

    Args:
        report_id: The report to get executions for
        limit: Maximum number of executions to return

    Returns:
        List of ReportExecution objects ordered by most recent first
    """
    session = SessionLocal()
    try:
        return (
            session.query(ReportExecution)
            .filter(ReportExecution.report_id == report_id)
            .order_by(ReportExecution.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()


def get_latest_execution(report_id: int) -> ReportExecution | None:
    """Get the most recent execution for a report."""
    session = SessionLocal()
    try:
        return (
            session.query(ReportExecution)
            .filter(ReportExecution.report_id == report_id)
            .order_by(ReportExecution.created_at.desc())
            .first()
        )
    finally:
        session.close()


# ==============================================================================
# Automatic Report Detection
# ==============================================================================


def find_matching_report(user_id: int, question: str, sql: str = None) -> SavedReport | None:
    """Check if a question/SQL matches an existing saved report.

    Uses fuzzy matching (≥90% similarity) against:
    - original_question field
    - sql_query field (if provided)

    Args:
        user_id: The user whose reports to check
        question: The user's question
        sql: Optional SQL query to also compare

    Returns:
        The matching SavedReport or None if no match found
    """
    if not question:
        return None

    session = SessionLocal()
    try:
        reports = (
            session.query(SavedReport)
            .filter(SavedReport.user_id == user_id, SavedReport.is_archived == False)  # noqa: E712
            .all()
        )

        best_match = None
        best_score = 0

        for report in reports:
            # Check question similarity
            q_score = _calculate_similarity(question, report.original_question)

            # Check SQL similarity if provided
            sql_score = 0
            if sql and report.sql_query:
                sql_score = _calculate_similarity(sql, report.sql_query)

            # Use the higher of the two scores
            score = max(q_score, sql_score)

            if score >= SIMILARITY_THRESHOLD and score > best_score:
                best_score = score
                best_match = report

        if best_match:
            logger.debug(
                f"Found matching report '{best_match.name}' (id={best_match.id}) "
                f"with similarity {best_score:.2%}"
            )

        return best_match

    finally:
        session.close()


# ==============================================================================
# Trend Analysis
# ==============================================================================


def get_report_trends(report_id: int, column: str = None, metric: str = "sum") -> list[dict]:
    """Get trend data for a report across all executions.

    Args:
        report_id: The report to get trends for
        column: Specific column to get trends for (or None for all)
        metric: The metric to extract (sum, avg, min, max, count)

    Returns:
        List of dicts with {timestamp, column, value} for charting
    """
    session = SessionLocal()
    try:
        executions = (
            session.query(ReportExecution)
            .filter(ReportExecution.report_id == report_id, ReportExecution.success == True)  # noqa: E712
            .order_by(ReportExecution.created_at.asc())
            .all()
        )

        trends = []
        for exec in executions:
            if not exec.numeric_summary:
                continue

            try:
                summary = json.loads(exec.numeric_summary)
            except (json.JSONDecodeError, TypeError):
                continue

            if column:
                # Get specific column
                if column in summary and metric in summary[column]:
                    trends.append(
                        {
                            "timestamp": exec.created_at.isoformat() if exec.created_at else None,
                            "column": column,
                            "metric": metric,
                            "value": summary[column][metric],
                            "execution_id": exec.id,
                        }
                    )
            else:
                # Get all columns
                for col, metrics in summary.items():
                    if metric in metrics:
                        trends.append(
                            {
                                "timestamp": exec.created_at.isoformat() if exec.created_at else None,
                                "column": col,
                                "metric": metric,
                                "value": metrics[metric],
                                "execution_id": exec.id,
                            }
                        )

        return trends

    finally:
        session.close()


def get_consistent_columns(report_id: int) -> list[str]:
    """Return columns that exist in ALL executions for reliable trend charting.

    Args:
        report_id: The report to check

    Returns:
        List of column names present in every execution's numeric_summary
    """
    session = SessionLocal()
    try:
        executions = (
            session.query(ReportExecution)
            .filter(ReportExecution.report_id == report_id, ReportExecution.success == True)  # noqa: E712
            .all()
        )

        if not executions:
            return []

        # Start with columns from first execution
        consistent_cols = None
        for exec in executions:
            if not exec.numeric_summary:
                continue
            try:
                summary = json.loads(exec.numeric_summary)
                cols = set(summary.keys())
                if consistent_cols is None:
                    consistent_cols = cols
                else:
                    consistent_cols &= cols
            except (json.JSONDecodeError, TypeError):
                continue

        return list(consistent_cols) if consistent_cols else []

    finally:
        session.close()


def compare_executions(exec1_id: int, exec2_id: int) -> pd.DataFrame | None:
    """Compare two executions side by side.

    Args:
        exec1_id: First execution (typically older)
        exec2_id: Second execution (typically newer)

    Returns:
        DataFrame with comparison or None if either execution not found
    """
    session = SessionLocal()
    try:
        exec1 = session.query(ReportExecution).filter(ReportExecution.id == exec1_id).first()
        exec2 = session.query(ReportExecution).filter(ReportExecution.id == exec2_id).first()

        if not exec1 or not exec2:
            return None

        if not exec1.numeric_summary or not exec2.numeric_summary:
            return None

        try:
            summary1 = json.loads(exec1.numeric_summary)
            summary2 = json.loads(exec2.numeric_summary)
        except (json.JSONDecodeError, TypeError):
            return None

        # Build comparison data
        all_cols = set(summary1.keys()) | set(summary2.keys())
        rows = []
        for col in sorted(all_cols):
            for metric in ["sum", "avg", "min", "max", "count"]:
                val1 = summary1.get(col, {}).get(metric)
                val2 = summary2.get(col, {}).get(metric)
                if val1 is not None or val2 is not None:
                    delta = None
                    delta_pct = None
                    if val1 is not None and val2 is not None and val1 != 0:
                        delta = val2 - val1
                        delta_pct = (delta / val1) * 100

                    rows.append(
                        {
                            "Column": col,
                            "Metric": metric,
                            "Previous": val1,
                            "Current": val2,
                            "Delta": delta,
                            "Delta %": delta_pct,
                        }
                    )

        return pd.DataFrame(rows) if rows else None

    finally:
        session.close()


def get_execution_count(report_id: int) -> int:
    """Get the total number of executions for a report."""
    session = SessionLocal()
    try:
        return session.query(ReportExecution).filter(ReportExecution.report_id == report_id).count()
    finally:
        session.close()


def format_delta(current: float, previous: float) -> str:
    """Format a delta value as a string with arrow indicator.

    Args:
        current: Current value
        previous: Previous value

    Returns:
        Formatted string like "↑ 12.5%" or "↓ 3.2%" or "→ 0%"
    """
    if previous == 0:
        return "N/A"

    delta_pct = ((current - previous) / previous) * 100

    if delta_pct > 0:
        return f"↑ {delta_pct:.1f}%"
    elif delta_pct < 0:
        return f"↓ {abs(delta_pct):.1f}%"
    else:
        return "→ 0%"
