"""SQLite-backed section cache for Patient 360 (#246).

Persists one summary per (source_id, section) in ``thrive_patient360_section``,
keyed by a versioned content fingerprint. ``get`` returns a stored narrative
only when the fingerprint matches (input unchanged, same generator); otherwise
the section is regenerated and re-stored. Implements the ``SectionCache``
protocol in ``agent.patient360.pipeline``.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.patient360.prompts import GENERATOR_VERSION


class SqlitePatient360Cache:
    def __init__(self, session: Any):
        self._session = session

    def get(self, source_id: str, section: str, fingerprint: str) -> Optional[str]:
        from orm.models import Patient360Section

        row = self._session.query(Patient360Section).filter_by(source_id=source_id, section=section).one_or_none()
        if row is not None and row.fingerprint == fingerprint:
            return row.narrative
        return None

    def put(self, source_id: str, section: str, fingerprint: str, narrative: str) -> None:
        from orm.models import Patient360Section

        row = self._session.query(Patient360Section).filter_by(source_id=source_id, section=section).one_or_none()
        if row is None:
            row = Patient360Section(source_id=source_id, section=section)
            self._session.add(row)
        row.fingerprint = fingerprint
        row.generator_version = GENERATOR_VERSION
        row.narrative = narrative
        self._session.commit()
