"""Consent SQL against the HealtheLink warehouse (#242 / #243 / #244).

Source of truth (thrive decision, 2026-06-23): consent lives on
``federated_demographic_v.hie_consent`` — NOT membership in
``internal_patient_profile_v`` (the ~33% no-consent finding, #242, disproves
that shortcut). ``hie_consent`` only ever holds ``'TRUE'``, ``'FALSE'``, or the
empty string; blank is *unknown*, not a third consent state (ADT feeds are
blank-only), so it is filtered out with ``NULLIF(TRIM(hie_consent), '')`` and
never treated as consent.

A patient federates across sibling source_ids, and consent can be re-stated
over time, so authority is the **latest explicit** value by recency, taken
across the patient's whole federation set. Fail-closed: only ``'TRUE'`` grants;
``'FALSE'``, blank-only, or no row at all denies.

NOTE (flagged for HeL/Sarah sign-off): the recency column is assumed to be
``created_date`` (present on the view); the meeting notes reference a
``last_modified_date_time``. If the authoritative view exposes a distinct
last-modified timestamp, swap ``_RECENCY_COL``. The "latest wins" reading is
revocation-aware (a later FALSE overrides an earlier TRUE); the notes also say
"once given, consent is lifelong" — these conflict, and latest-wins is the
safer fail-closed choice until HeL confirms.
"""

from __future__ import annotations

from typing import Tuple

# Recency column used to order a patient's consent rows. See module note.
_RECENCY_COL = "created_date"

# The literal a suppressed / non-consented lookup must be indistinguishable
# from — consent status is never inferable by comparing responses.
CONSENT_GRANTED_VALUE = "TRUE"


def _sid_placeholders(source_ids: list[str]) -> Tuple[str, dict]:
    """Build a portable ``IN (...)`` clause + params for a source_id list."""
    keys = [f"sid_{i}" for i in range(len(source_ids))]
    params = {k: s for k, s in zip(keys, source_ids)}
    placeholders = ", ".join(f":{k}" for k in keys)
    return placeholders, params


def latest_consent_sql(*, source_ids: list[str], schema_prefix: str = "") -> Tuple[str, dict]:
    """Return (sql, params) for the latest EXPLICIT consent across source_ids.

    Yields at most one row: the most recent non-blank ``hie_consent`` value for
    any of the patient's federated source_ids. No row => never explicitly
    consented => deny. Caller compares the value to ``CONSENT_GRANTED_VALUE``.

    An empty ``source_ids`` is refused rather than issuing an unbounded scan —
    the caller (an ambiguous/unknown patient) must already be failing closed.
    """
    if not source_ids:
        raise ValueError("latest_consent_sql requires at least one source_id.")
    placeholders, params = _sid_placeholders(source_ids)
    sql = f"""
    SELECT UPPER(TRIM(hie_consent)) AS hie_consent
    FROM {schema_prefix}federated_demographic_v
    WHERE source_id IN ({placeholders})
      AND NULLIF(TRIM(hie_consent), '') IS NOT NULL
    ORDER BY {_RECENCY_COL} DESC
    LIMIT 1
    """
    return sql, params


def consent_population_counts_sql(*, schema_prefix: str = "") -> Tuple[str, dict]:
    """Count-only population consent breakdown for the #242 investigation.

    Buckets every row of ``federated_demographic_v`` by explicit consent state
    (TRUE / FALSE / blank-or-null). Returns aggregates only — no identifiers,
    no patient rows — so it is safe to run against prod under the PHI rules.
    Run per-patient-grain by wrapping the caller's own distinct-patient logic;
    at row grain this measures the raw distribution the ~33% finding came from.
    """
    sql = f"""
    SELECT
        SUM(CASE WHEN UPPER(TRIM(hie_consent)) = 'TRUE'  THEN 1 ELSE 0 END) AS consent_true,
        SUM(CASE WHEN UPPER(TRIM(hie_consent)) = 'FALSE' THEN 1 ELSE 0 END) AS consent_false,
        SUM(CASE WHEN NULLIF(TRIM(hie_consent), '') IS NULL THEN 1 ELSE 0 END) AS consent_blank,
        COUNT(*) AS total_rows
    FROM {schema_prefix}federated_demographic_v
    """
    return sql, {}
