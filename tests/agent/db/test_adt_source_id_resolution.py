"""Behavioral tests: admissions_sql must resolve the entered source_id at ANY
empi_rank, not only rank 1.

Background: the per-patient admissions query resolved identity with
`isr.source_id = :source_id AND isr.empi_rank = 1`, so a patient selected
under a sibling/old CID (any non-rank-1 id) — or any patient with no rank-1
xref row at all — silently got "no records found" for admissions while every
other domain (federated fan-out) worked. The fix is a best-rank resolver
subquery: resolve the entered id to its best-ranked patient_id mapping
(lowest empi_rank wins, NULLs pinned last via COALESCE, patient_id as a
deterministic tie-break), exactly one resolver row, then join ADT rows on
patient_id as before.

The sibling-CID case also runs against the shared synthetic fixture
(src-john-1962-alt, empi_rank=2) in this file's synthetic_db tests.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.adt import admissions_sql


# --------------------------------------------------------------------------
# Standalone SQLite fixture (covers cases the shared synthetic db lacks:
# a patient with NO rank-1 row, and duplicate xref rows for one source_id)
# --------------------------------------------------------------------------

# (patient_id, source_id, empi_rank, source_name)
_SOURCE_REFS = [
    (1, "P1-RANK1", 1, "EHR-A"),
    (1, "P1-RANK1", 1, "EHR-C"),  # same source_id recurs per (source_id, source_name)
    (1, "P1-RANK2", 2, "EHR-B"),
    (2, "P2-RANK2-ONLY", 2, "EHR-A"),  # patient 2 has NO empi_rank=1 row
]

# (patient_id, visit_number, event_date, clean_status, clean_setting)
_ADT_EVENTS = [
    ("1", "V1", "2025-06-15 08:00", "ADMIT", "INPATIENT"),
    ("1", "V1", "2025-06-18 10:00", "DISCHARGE", "INPATIENT"),
    ("2", "V2", "2026-01-10 09:00", "ADMIT", "INPATIENT"),
]


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://")
    with eng.begin() as c:
        c.execute(
            text(
                "CREATE TABLE internal_source_reference_v ("
                "patient_id INTEGER, source_id TEXT, empi_rank INTEGER, source_name TEXT)"
            )
        )
        c.execute(
            text(
                "CREATE TABLE federated_adt_v ("
                "patient_id TEXT, visit_number TEXT, event_date TEXT, "
                "clean_status TEXT, clean_setting TEXT, cancelled_flag TEXT, "
                "event_location TEXT, location_type TEXT, admit_from TEXT, "
                "diagnosing_clinician TEXT, "
                "discharge_disposition TEXT, discharge_location TEXT)"
            )
        )
        for pid, sid, rank, src in _SOURCE_REFS:
            c.execute(
                text(
                    "INSERT INTO internal_source_reference_v "
                    "(patient_id, source_id, empi_rank, source_name) "
                    "VALUES (:pid, :sid, :rank, :src)"
                ),
                {"pid": pid, "sid": sid, "rank": rank, "src": src},
            )
        for pid, visit, dt, status, setting in _ADT_EVENTS:
            c.execute(
                text(
                    "INSERT INTO federated_adt_v "
                    "(patient_id, visit_number, event_date, clean_status, clean_setting, "
                    "cancelled_flag, event_location, location_type, admit_from, "
                    "discharge_disposition, discharge_location) "
                    "VALUES (:pid, :visit, :dt, :status, :setting, 'N', "
                    "'Hosp', 'Hospital', NULL, NULL, NULL)"
                ),
                {"pid": pid, "visit": visit, "dt": dt, "status": status, "setting": setting},
            )
    yield eng
    eng.dispose()


def _rows(engine, source_id, **kwargs):
    sql, params = admissions_sql(source_id=source_id, dialect="sqlite", facility_type="any", **kwargs)
    with engine.connect() as c:
        return c.execute(text(sql), params).mappings().all()


class TestAnyRankResolution:
    def test_rank_1_id_still_returns_admissions(self, engine):
        rows = _rows(engine, "P1-RANK1")
        assert {r["visit_number"] for r in rows} == {"V1"}

    def test_sibling_rank_2_id_returns_the_same_admissions(self, engine):
        """Before the fix: [] — the rank-1-only join found no xref row."""
        rows = _rows(engine, "P1-RANK2")
        assert {r["visit_number"] for r in rows} == {"V1"}

    def test_patient_without_rank_1_row_gets_admissions(self, engine):
        """Before the fix, patient 2 (empi_rank=2 only) could NEVER see admissions."""
        rows = _rows(engine, "P2-RANK2-ONLY")
        assert {r["visit_number"] for r in rows} == {"V2"}

    def test_duplicate_xref_rows_do_not_duplicate_visits(self, engine):
        """internal_source_reference_v holds one row per (source_id, source_name);
        the resolver must collapse them to exactly one row."""
        rows = _rows(engine, "P1-RANK1")
        assert len(rows) == 1

    def test_output_source_id_is_the_entered_id(self, engine):
        assert all(r["source_id"] == "P1-RANK2" for r in _rows(engine, "P1-RANK2"))

    def test_unknown_source_id_returns_nothing(self, engine):
        assert _rows(engine, "NO-SUCH-CID") == []


# --------------------------------------------------------------------------
# Shared synthetic fixture: the realistic sibling-CID case
# --------------------------------------------------------------------------


def test_sibling_cid_sees_same_visits_as_canonical(synthetic_db):
    """src-john-1962-alt is john's empi_rank=2 CID; admissions must match the
    canonical id's result instead of coming back empty."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    canonical_sql, canonical_params = admissions_sql(source_id="src-john-1962", dialect="sqlite", facility_type="any")
    sibling_sql, sibling_params = admissions_sql(source_id="src-john-1962-alt", dialect="sqlite", facility_type="any")
    canonical = adapter.fetch_all(canonical_sql, canonical_params)
    sibling = adapter.fetch_all(sibling_sql, sibling_params)
    assert {r["visit_number"] for r in canonical} == {"V100", "V101", "V102"}
    assert {r["visit_number"] for r in sibling} == {"V100", "V101", "V102"}
