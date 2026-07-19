"""Consented-patient roster snapshot (#317) — Chiron's snapshot approach.

The live ConsentGate runs the union-contract CTE per patient check (scans both
demographic views before scoping). That's correct but O(warehouse) per lookup,
and find_patient gates every result row. The snapshot materializes the
consented population once — every ``patient_id`` whose latest explicit consent
is TRUE — into an indexed set, so a gate check becomes a point lookup.

This module provides the in-memory snapshot + its build query. A durable,
atomically-refreshed table + a consent-feed watermark (refresh only when the
HeL feed advanced; ADT/claims must not move it) are the remaining #317 work —
tracked as follow-up; the point-lookup contract here is what they'll back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Optional

from agent.db.queries.consent import consented_roster_sql


@dataclass(frozen=True)
class ConsentSnapshot:
    """Immutable point-in-time set of consented EMPI patient_ids."""

    patient_ids: FrozenSet[int]
    built_at: Optional[str] = None  # ISO timestamp, stamped by the builder/refresh

    def is_consented(self, patient_id: Optional[int]) -> bool:
        """Fail-closed point lookup: True only if patient_id is in the roster."""
        return patient_id is not None and patient_id in self.patient_ids

    @classmethod
    def build(cls, adapter: Any, *, schema_prefix: str = "", built_at: Optional[str] = None) -> "ConsentSnapshot":
        """Materialize the roster from the warehouse (the union contract, once)."""
        dialect = getattr(adapter, "dialect", "sqlite")
        sql, params = consented_roster_sql(schema_prefix=schema_prefix, dialect=dialect)
        rows = adapter.fetch_all(sql, params)
        return cls(patient_ids=frozenset(r["patient_id"] for r in rows), built_at=built_at)
