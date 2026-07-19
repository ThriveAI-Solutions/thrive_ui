"""Entry-point orchestration for Patient 360 (#246).

Builds the analytics adapter + model-backed summarizer and runs the pipeline for
a source_id. Kept separate from the Streamlit view so it is unit-testable with
an injected adapter/summarizer.
"""

from __future__ import annotations

from typing import Any, List, Optional

from agent.patient360.pipeline import Patient360Result, Summarizer, generate_patient360


def run_patient360(
    source_id: str,
    *,
    adapter: Any = None,
    schema_prefix: Optional[str] = None,
    summarizer: Optional[Summarizer] = None,
    sections: Optional[List[str]] = None,
) -> Patient360Result:
    """Run Patient 360 for ``source_id``.

    ``adapter`` / ``summarizer`` default to the app-configured warehouse adapter
    and model (built lazily); inject them in tests. ``schema_prefix`` defaults to
    the adapter's own prefix.
    """
    if adapter is None:
        from agent.db.analytics_adapter import AnalyticsDbAdapter

        adapter = AnalyticsDbAdapter.from_streamlit_secrets()
    if schema_prefix is None:
        schema_prefix = getattr(adapter, "schema_prefix", "")
    if summarizer is None:
        from agent.patient360.summarizer import build_patient360_summarizer

        summarizer = build_patient360_summarizer()

    return generate_patient360(
        adapter, source_id, schema_prefix=schema_prefix, summarizer=summarizer, sections=sections
    )
