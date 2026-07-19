"""Consent snapshot roster + point-lookup gate (#317)."""

from sqlalchemy import create_engine, text

from agent.consent.gate import ConsentGate
from agent.consent.snapshot import ConsentSnapshot
from agent.db.analytics_adapter import AnalyticsDbAdapter


def _adapter(rows):
    """rows: (patient_id, source_id, hie_consent) — one demographic row each,
    linked at empi_rank 1."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as c:
        c.execute(
            text("CREATE TABLE internal_source_reference_v (patient_id INTEGER, source_id TEXT, empi_rank INTEGER)")
        )
        for t in ("federated_demographic_v", "federated_demographic_history_v"):
            c.execute(
                text(
                    f"CREATE TABLE {t} (source_id TEXT, source_name TEXT, "
                    "hie_consent TEXT, last_modified_datetime TEXT)"
                )
            )
        for pid, sid, consent in rows:
            c.execute(
                text("INSERT INTO internal_source_reference_v VALUES (:p, :s, 1)"),
                {"p": pid, "s": sid},
            )
            c.execute(
                text(
                    "INSERT INTO federated_demographic_v "
                    "(source_id, source_name, hie_consent, last_modified_datetime) "
                    "VALUES (:s, 'SRC', :c, '2026-01-01')"
                ),
                {"s": sid, "c": consent},
            )
    return AnalyticsDbAdapter(engine=eng, dialect="sqlite")


def test_roster_contains_only_consented_patients():
    adapter = _adapter([(1, "s1", "TRUE"), (2, "s2", "FALSE"), (3, "s3", "TRUE")])
    snap = ConsentSnapshot.build(adapter, built_at="2026-07-19T00:00:00Z")
    assert snap.patient_ids == frozenset({1, 3})
    assert snap.built_at == "2026-07-19T00:00:00Z"


def test_snapshot_point_lookup_is_fail_closed():
    snap = ConsentSnapshot(patient_ids=frozenset({1, 3}))
    assert snap.is_consented(1) is True
    assert snap.is_consented(2) is False  # not consented
    assert snap.is_consented(99) is False  # unknown
    assert snap.is_consented(None) is False


def test_gate_uses_snapshot_when_present():
    # Adapter would raise if queried (no consent tables) — proves the gate uses
    # the snapshot point lookup, not the live CTE.
    from unittest.mock import MagicMock

    boom = MagicMock()
    boom.dialect = "sqlite"
    boom.fetch_all.side_effect = AssertionError("live query must not run when a snapshot is set")
    snap = ConsentSnapshot(patient_ids=frozenset({5}))
    gate = ConsentGate(boom, snapshot=snap)
    assert gate.is_consented(5) is True
    assert gate.is_consented(6) is False
    boom.fetch_all.assert_not_called()


def test_gate_falls_back_to_live_query_without_snapshot():
    adapter = _adapter([(7, "s7", "TRUE"), (8, "s8", "FALSE")])
    gate = ConsentGate(adapter)  # no snapshot -> live union-contract query
    assert gate.is_consented(7) is True
    assert gate.is_consented(8) is False


def test_snapshot_matches_live_gate_decisions():
    adapter = _adapter([(1, "s1", "TRUE"), (2, "s2", "FALSE"), (3, "s3", "TRUE")])
    snap = ConsentSnapshot.build(adapter)
    live = ConsentGate(adapter)
    snap_gate = ConsentGate(adapter, snapshot=snap)
    for pid in (1, 2, 3, 99):
        assert snap_gate.is_consented(pid) == live.is_consented(pid)
