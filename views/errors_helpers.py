"""Pure helpers for the Errors admin page (views/errors.py).

Split out so unit tests can import them without triggering the Streamlit
script that lives at the bottom of ``views/errors.py``. No Streamlit
imports in this module — only stdlib, pandas, and the read model.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from orm.error_read_model import (
    DEFAULT_SOURCES,
    ErrorRow,
    ErrorSource,
    count_errors_by_source,
    query_errors,
)
from utils.error_fallback_sink import ErrorLoggingConfig

_RESULT_COLUMNS = (
    "created_at",
    "source",
    "category",
    "severity",
    "error_type",
    "error_message",
)


def _time_range_to_since(days: int) -> dt.datetime:
    """Return the ``since`` cutoff for a 'last N days' filter."""
    return dt.datetime.now() - dt.timedelta(days=days)


def _resolve_sources(csv: str) -> set[ErrorSource]:
    """Parse the comma-separated source value list used as a cache key.

    Empty string → :data:`DEFAULT_SOURCES`. Unknown / malformed values
    are silently dropped so a corrupted cache key cannot crash the page.
    """
    if not csv or not csv.strip():
        return set(DEFAULT_SOURCES)
    out: set[ErrorSource] = set()
    for piece in csv.split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            out.add(ErrorSource(token))
        except ValueError:
            continue
    return out


def _csv_to_set(csv: str) -> set[str] | None:
    """Parse a comma-separated filter string into a set or ``None``.

    Empty / whitespace-only input returns ``None`` (which the read-model
    adapters interpret as "no filter"). Per-value whitespace is stripped.
    """
    if not csv or not csv.strip():
        return None
    tokens = {piece.strip() for piece in csv.split(",") if piece.strip()}
    return tokens or None


def _parse_user_id(text: str) -> int | None:
    """Parse a user-id text input into ``int | None``.

    Empty input, whitespace-only, non-numeric, or negative values all
    return ``None`` so the filter degenerates to "no user filter" rather
    than raising.
    """
    if not text or not text.strip():
        return None
    try:
        value = int(text.strip())
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def _export_filename(
    since: dt.datetime,
    sources_csv: str,
    until: dt.datetime | None = None,
) -> str:
    """Build a descriptive filename for the JSON export download.

    Format: ``errors_<since>_<until>_<sources>.json``. Empty
    ``sources_csv`` becomes ``"all"``.
    """
    since_iso = since.strftime("%Y%m%dT%H%M%S")
    until_iso = (until or dt.datetime.now()).strftime("%Y%m%dT%H%M%S")
    src_summary = sources_csv.replace(",", "-") if sources_csv else "all"
    return f"errors_{since_iso}_{until_iso}_{src_summary}.json"


def _pretty_context_data(text: str | None) -> str | None:
    """Pretty-print a JSON-string ``context_data`` field for the drill-down.

    Returns the input unchanged if it's not valid JSON (so partial JSON
    or other formats are still visible to the admin). ``None`` and empty
    strings round-trip as ``None``.
    """
    if not text or not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text
    try:
        return json.dumps(parsed, indent=2, default=str)
    except (TypeError, ValueError):
        return text


def _query_filtered(
    days: int,
    sources_csv: str,
    categories_csv: str,
    severities_csv: str,
    search: str,
    user_id_text: str,
    *,
    fallback_config: ErrorLoggingConfig | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Compose the view's data load from stringly-typed widget inputs.

    Returns ``(rows_as_dicts, counts_by_source_value)``. Counts are
    deliberately unfiltered so the per-source badges show the full
    available total under the selected time range.
    """
    since = _time_range_to_since(days)
    selected = _resolve_sources(sources_csv) or set(DEFAULT_SOURCES)
    rows = query_errors(
        since,
        sources=selected,
        categories=_csv_to_set(categories_csv),
        severities=_csv_to_set(severities_csv),
        user_id=_parse_user_id(user_id_text),
        search=search.strip() or None,
        fallback_config=fallback_config,
    )
    counts = count_errors_by_source(since, fallback_config=fallback_config)
    return (
        [r.to_dict() for r in rows],
        {src.value: n for src, n in counts.items()},
    )


def _format_results_table(rows: list[ErrorRow]) -> pd.DataFrame:
    """Build the basic results table shown above the per-row drill-down.

    Empty input returns a DataFrame with the expected columns but no
    rows so downstream Streamlit code can safely render an empty grid.
    None values on optional columns become empty strings for display.
    """
    if not rows:
        return pd.DataFrame(columns=list(_RESULT_COLUMNS))
    return pd.DataFrame(
        {
            "created_at": [r.created_at for r in rows],
            "source": [r.source.value for r in rows],
            "category": [r.category or "" for r in rows],
            "severity": [r.severity or "" for r in rows],
            "error_type": [r.error_type or "" for r in rows],
            "error_message": [r.error_message or "" for r in rows],
        }
    )
