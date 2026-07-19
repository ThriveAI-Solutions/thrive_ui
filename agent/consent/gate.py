"""Fail-closed per-patient consent gate (#244 CON-4).

One decision point: given a patient's federated source_ids, is that patient
currently consented to be disclosed? Every failure mode denies — no source_ids
(unknown/ambiguous patient), no explicit consent row, a non-TRUE latest value,
or any error querying consent. A denied patient is surfaced with the exact same
"not found" response as a nonexistent or ambiguous one (see
``agent.db.federation`` and ``find_patient``), so a caller can never distinguish
"exists but not consented" from "does not exist."

Consent source of truth is ``federated_demographic_v.hie_consent`` (latest
explicit value) — see ``agent.db.queries.consent``.

Role-based bypass (flagged, NOT wired): the thrive meeting notes describe an
Erie-County-clinical role that bypasses consent while other roles see only
consented patients. ``consent_required(role)`` is the single hook for that
policy; it currently returns True for every role (consent enforced for all)
until the role->policy mapping is ratified and wired. Do NOT scatter role
checks elsewhere — extend this function.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.db.queries.consent import CONSENT_GRANTED_VALUE, patient_consent_sql
from utils.quick_logger import get_logger

logger = get_logger(__name__)


def consent_required(role: Optional[str]) -> bool:
    """Whether consent enforcement applies to this role.

    Default: True for every role. This is the sole place to later exempt the
    Erie-County-clinical role (thrive notes) once HeL ratifies the mapping.
    """
    return True


class ConsentGate:
    """Fail-closed consent check over the analytics warehouse.

    ``enforcing=False`` makes the gate allow everything — a kill switch for
    environments/tests that have no consent data, never a silent default. In
    production the gate is constructed enforcing; an enforcing adapter with no
    gate at all must itself deny (the caller's responsibility).
    """

    def __init__(self, adapter: Any, *, schema_prefix: str = "", enforcing: bool = True):
        self._adapter = adapter
        self._schema_prefix = schema_prefix
        self._enforcing = enforcing

    def is_consented(self, patient_id: Optional[int]) -> bool:
        """True only if the EMPI patient's latest explicit consent is TRUE.

        Consent is person-grain (per internal patient_id), computed by the
        authoritative union contract in agent.db.queries.consent. Fail-closed on
        every other outcome: no patient_id (unknown/ambiguous), no explicit
        event, a FALSE latest value, or a query error.
        """
        if not self._enforcing:
            return True
        if patient_id is None:
            return False
        dialect = getattr(self._adapter, "dialect", "sqlite")
        try:
            sql, params = patient_consent_sql(patient_id=patient_id, schema_prefix=self._schema_prefix, dialect=dialect)
            rows = self._adapter.fetch_all(sql, params)
        except Exception:
            # Any failure denies — never let a query error open the gate.
            logger.warning("Consent lookup failed; denying fail-closed.", exc_info=True)
            return False
        if not rows:
            return False
        return str(rows[0].get("consent") or "").strip().upper() == CONSENT_GRANTED_VALUE
