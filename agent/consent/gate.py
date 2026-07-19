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

Role-based bypass: the thrive meeting notes ratify the *principle* — an
Erie-County-clinical role bypasses consent while every other role sees only
consented patients — but NOT an enumerated per-role table, and the generic
Okta roles are admin/doctor/nurse/patient (no distinct Erie-County-clinical
enum member yet). So the bypass set is data-driven, sourced from
``[security].consent_bypass_roles`` (parsed by ``parse_bypass_roles``) and
threaded onto ``AgentDeps``. It defaults EMPTY — consent enforced for every
role — so nothing bypasses until HeL ratifies the mapping and an operator
sets it. ``consent_required(role, bypass_roles)`` is the single hook; do NOT
scatter role checks elsewhere — extend this function.
"""

from __future__ import annotations

from typing import Any, FrozenSet, Iterable, Optional

from agent.db.queries.consent import CONSENT_GRANTED_VALUE, patient_consent_sql
from orm.models import RoleTypeEnum
from utils.quick_logger import get_logger

logger = get_logger(__name__)


def consent_required(role: Optional[RoleTypeEnum], bypass_roles: FrozenSet[RoleTypeEnum] = frozenset()) -> bool:
    """Whether consent enforcement applies to this role.

    Returns False (exempt) only when ``role`` is in the ratified ``bypass_roles``
    set; True otherwise. ``bypass_roles`` defaults empty, so with no ratified
    mapping wired every role — including an unknown/None one — requires consent.
    This is the sole place to exempt the Erie-County-clinical role once HeL
    ratifies the mapping.
    """
    return role not in bypass_roles


def parse_bypass_roles(raw: Optional[Iterable[Any]]) -> FrozenSet[RoleTypeEnum]:
    """Parse a config list into a set of consent-bypass roles, fail-safe.

    Accepts role names ("admin", case-insensitive) or int values (0..3), or
    ``RoleTypeEnum`` members. Anything unrecognized is DROPPED (never added):
    a typo or an unknown role must not silently open the gate — the safe
    failure is "consent still enforced for that role." ``None``/empty -> empty.
    """
    if not raw:
        return frozenset()
    out: set[RoleTypeEnum] = set()
    for item in raw:
        if isinstance(item, RoleTypeEnum):
            out.add(item)
            continue
        if isinstance(item, bool):  # bool is an int subclass; never a role code
            continue
        if isinstance(item, int):
            try:
                out.add(RoleTypeEnum(item))
            except ValueError:
                logger.warning("Ignoring unknown consent_bypass_roles code: %r", item)
            continue
        if isinstance(item, str):
            try:
                out.add(RoleTypeEnum[item.strip().upper()])
            except KeyError:
                logger.warning("Ignoring unknown consent_bypass_roles name: %r", item)
            continue
        logger.warning("Ignoring unparseable consent_bypass_roles entry: %r", item)
    return frozenset(out)


class ConsentGate:
    """Fail-closed consent check over the analytics warehouse.

    ``enforcing=False`` makes the gate allow everything — a kill switch for
    environments/tests that have no consent data, never a silent default. In
    production the gate is constructed enforcing; an enforcing adapter with no
    gate at all must itself deny (the caller's responsibility).
    """

    def __init__(self, adapter: Any, *, schema_prefix: str = "", enforcing: bool = True, snapshot: Any = None):
        self._adapter = adapter
        self._schema_prefix = schema_prefix
        self._enforcing = enforcing
        # Optional ConsentSnapshot (#317): when present, is_consented is a
        # point lookup against the materialized roster instead of the live
        # per-patient union-contract query.
        self._snapshot = snapshot

    def is_consented(self, patient_id: Optional[int]) -> bool:
        """True only if the EMPI patient's latest explicit consent is TRUE.

        Consent is person-grain (per internal patient_id), computed by the
        authoritative union contract in agent.db.queries.consent — or, when a
        snapshot is supplied, a point lookup against it (#317). Fail-closed on
        every other outcome: no patient_id (unknown/ambiguous), no explicit
        event, a FALSE latest value, or a query error.
        """
        if not self._enforcing:
            return True
        if patient_id is None:
            return False
        if self._snapshot is not None:
            return bool(self._snapshot.is_consented(patient_id))
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
