"""Consent SQL against the HealtheLink warehouse (#242 / #243 / #244).

Implements the authoritative post-Fusion consent contract (Sarah/HeL, via Joe):

- Consent events are explicit ``hie_consent`` values ('TRUE'/'FALSE'; blank is
  unknown, filtered out) drawn from BOTH ``federated_demographic_v`` and
  ``federated_demographic_history_v``.
- Claims-sourced rows (``source_name = 'claims_process'``) carry a 2-part
  ``payer-member-`` prefix on their source_id, so a raw join reaches only a
  fraction of claims consent. Those rows are re-joined on a normalized key that
  strips the prefix (the "union contract").
- Events link to an EMPI person via ``internal_source_reference_v`` (excluding
  stale rank 99 and the ``patient_id = -1`` orphan).
- Per person, the LATEST explicit event by ``last_modified_datetime`` wins;
  FALSE wins ties (revocation-safe). Consent is therefore person-grain
  (per internal ``patient_id``), not per source_id.

``patient_consent_sql`` scopes this to one patient for the live gate;
``consent_population_counts_sql`` runs it whole for the #242 investigation and
adds a ``NEVER_EXPLICIT`` bucket (profiled patients with no explicit event) —
that bucket is where the ~33%-no-consent finding lives.

NOTE (perf): the live per-patient query still scans the demographic views
before scoping in the join. A materialized person-grain snapshot (indexed by
patient_id, atomically refreshed) is the scalable form — tracked as follow-up.
"""

from __future__ import annotations

from typing import Tuple

CONSENT_GRANTED_VALUE = "TRUE"


def _claims_join_key(dialect: str) -> str:
    """Expression that strips the 2-part ``payer-member-`` claims prefix."""
    if dialect in ("postgres", "redshift"):
        return "NULLIF(REGEXP_REPLACE(source_id, '^[^-]*-[^-]*-', ''), '')"
    # SQLite has no REGEXP_REPLACE: strip through the 2nd '-' with SUBSTRING/INSTR.
    # SUBSTRING (not SUBSTR — Redshift rejects that spelling; guarded by
    # test_redshift_compatibility) is a SQLite alias; this branch is sqlite-only.
    return (
        "NULLIF(SUBSTRING(source_id, INSTR(source_id, '-') + "
        "INSTR(SUBSTRING(source_id, INSTR(source_id, '-') + 1), '-') + 1), '')"
    )


def _person_latest_cte(schema_prefix: str, dialect: str, *, patient_scope: bool) -> str:
    """WITH consent_ev, linked, person_latest — the shared contract body.

    When ``patient_scope`` the linked join is filtered to ``:patient_id``.
    """
    claims_key = _claims_join_key(dialect)
    scope = "AND isr.patient_id = :patient_id" if patient_scope else ""
    return f"""
    WITH consent_ev AS (
        SELECT source_id, source_id AS join_key,
               UPPER(TRIM(hie_consent)) AS consent, last_modified_datetime
        FROM {schema_prefix}federated_demographic_v
        WHERE NULLIF(TRIM(hie_consent), '') IS NOT NULL
        UNION ALL
        SELECT source_id, source_id,
               UPPER(TRIM(hie_consent)), last_modified_datetime
        FROM {schema_prefix}federated_demographic_history_v
        WHERE NULLIF(TRIM(hie_consent), '') IS NOT NULL
        UNION ALL
        SELECT source_id, {claims_key},
               UPPER(TRIM(hie_consent)), last_modified_datetime
        FROM {schema_prefix}federated_demographic_v
        WHERE source_name = 'claims_process'
          AND NULLIF(TRIM(hie_consent), '') IS NOT NULL
        UNION ALL
        SELECT source_id, {claims_key},
               UPPER(TRIM(hie_consent)), last_modified_datetime
        FROM {schema_prefix}federated_demographic_history_v
        WHERE source_name = 'claims_process'
          AND NULLIF(TRIM(hie_consent), '') IS NOT NULL
    ),
    linked AS (
        SELECT DISTINCT isr.patient_id, ev.source_id, ev.consent, ev.last_modified_datetime
        FROM consent_ev ev
        JOIN {schema_prefix}internal_source_reference_v isr
          ON isr.source_id = ev.join_key
         AND isr.empi_rank <> 99
         AND CAST(isr.patient_id AS VARCHAR) <> '-1'
         {scope}
        WHERE ev.consent IN ('TRUE', 'FALSE')
    ),
    person_latest AS (
        SELECT patient_id, consent, last_modified_datetime AS consent_event_at
        FROM (
            SELECT patient_id, consent, last_modified_datetime,
                   ROW_NUMBER() OVER (
                       PARTITION BY patient_id
                       ORDER BY last_modified_datetime DESC NULLS LAST,
                                CASE WHEN consent = 'FALSE' THEN 0 ELSE 1 END ASC,
                                source_id ASC
                   ) AS rn
            FROM linked
        ) ranked WHERE rn = 1
    )"""


def patient_consent_sql(*, patient_id: int, schema_prefix: str = "", dialect: str = "sqlite") -> Tuple[str, dict]:
    """Latest explicit consent for ONE EMPI patient. At most one row; caller
    compares ``consent`` to ``CONSENT_GRANTED_VALUE``. No row => never
    explicitly consented => deny."""
    sql = (
        _person_latest_cte(schema_prefix, dialect, patient_scope=True)
        + """
    SELECT consent, consent_event_at FROM person_latest
    """
    )
    return sql, {"patient_id": patient_id}


def consented_roster_sql(*, schema_prefix: str = "", dialect: str = "sqlite") -> Tuple[str, dict]:
    """Every currently-consented EMPI patient_id (person_latest = 'TRUE').

    Source of truth for the consent snapshot (#317): materialize/point-look-up
    against this set instead of running the per-patient union-contract CTE on
    every gate check. Returns one column, patient_id.
    """
    sql = (
        _person_latest_cte(schema_prefix, dialect, patient_scope=False)
        + f"""
    SELECT patient_id FROM person_latest WHERE consent = '{CONSENT_GRANTED_VALUE}'
    """
    )
    return sql, {}


def consent_population_counts_sql(*, schema_prefix: str = "", dialect: str = "sqlite") -> Tuple[str, dict]:
    """Person-grain consent breakdown for the #242 investigation.

    Buckets every profiled patient as TRUE / FALSE (latest explicit) or
    NEVER_EXPLICIT (no explicit event at all). Aggregates only — safe to run
    against prod under the PHI rules. NEVER_EXPLICIT is the ~33% finding.
    """
    sql = (
        _person_latest_cte(schema_prefix, dialect, patient_scope=False)
        + f"""
    SELECT consent AS status, COUNT(*) AS n FROM person_latest GROUP BY consent
    UNION ALL
    SELECT 'NEVER_EXPLICIT' AS status, COUNT(*) AS n
    FROM {schema_prefix}internal_patient_profile_v prof
    WHERE CAST(prof.patient_id AS VARCHAR) <> '-1'
      AND NOT EXISTS (SELECT 1 FROM person_latest pl WHERE pl.patient_id = prof.patient_id)
    """
    )
    return sql, {}
