"""Entry-point orchestration for Patient 360 (#246).

Builds the analytics adapter + model-backed summarizer and runs the pipeline for
a source_id. Kept separate from the Streamlit view so it is unit-testable with
an injected adapter/summarizer.
"""

from __future__ import annotations

from typing import Any, List, Optional

from agent.patient360.pipeline import Patient360Result, SectionCache, Summarizer, generate_patient360


def _resolve_patient_id(adapter: Any, source_id: str, schema_prefix: str) -> Optional[int]:
    """Resolve source_id -> the single internal patient_id, or None if unknown
    or ambiguous (>1) — fail-closed for the consent check."""
    from agent.db.queries.patient import resolve_source_patient_sql

    sql, _ = resolve_source_patient_sql(schema_prefix=schema_prefix)
    rows = adapter.fetch_all(sql, {"source_id": source_id})
    pids = {r["internal_patient_id"] for r in rows}
    return next(iter(pids)) if len(pids) == 1 else None


def run_patient360(
    source_id: str,
    *,
    adapter: Any = None,
    schema_prefix: Optional[str] = None,
    summarizer: Optional[Summarizer] = None,
    sections: Optional[List[str]] = None,
    enforce_consent: bool = False,
    user_role: Any = None,
    cache: Optional[SectionCache] = None,
) -> Patient360Result:
    """Run Patient 360 for ``source_id``.

    ``adapter`` / ``summarizer`` default to the app-configured warehouse adapter
    and model (built lazily); inject them in tests. ``schema_prefix`` defaults to
    the adapter's own prefix.

    Consent (#244): when ``enforce_consent`` and the role requires it, the
    patient must be consented (person-grain gate) or the run fails closed to an
    EMPTY result — indistinguishable from a patient with no data, so consent is
    never inferable. Defense in depth: find_patient already omits non-consented
    patients from selection, but the 360 re-checks independently.
    """
    if adapter is None:
        from agent.db.analytics_adapter import AnalyticsDbAdapter

        adapter = AnalyticsDbAdapter.from_streamlit_secrets()
    if schema_prefix is None:
        schema_prefix = getattr(adapter, "schema_prefix", "")

    from agent.consent.gate import ConsentGate, consent_required

    if bool(enforce_consent) and consent_required(user_role):
        gate = ConsentGate(adapter, schema_prefix=schema_prefix, enforcing=True)
        patient_id = _resolve_patient_id(adapter, source_id, schema_prefix)
        if patient_id is None or not gate.is_consented(patient_id):
            return Patient360Result(source_id=source_id, sections=[], master_summary=None)

    if summarizer is None:
        from agent.patient360.summarizer import build_patient360_summarizer

        summarizer = build_patient360_summarizer()

    return generate_patient360(
        adapter, source_id, schema_prefix=schema_prefix, summarizer=summarizer, sections=sections, cache=cache
    )
