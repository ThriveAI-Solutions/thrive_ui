"""Unit + behavioral tests for the best-rank `internal_source_reference_v` join.

Background (2026-07-04 chiron prod audit): the cohort queries' hard
`AND isr.empi_rank = 1` inner join silently dropped 1,462 profiled patients
who have no rank-1 row in internal_source_reference_v. The fix is a
best-rank window join (lowest empi_rank wins, NULLs pinned last on every
dialect via COALESCE, source_id as a deterministic tie-break) that always
selects exactly one xref row per patient.

These tests need no warehouse: shape assertions are pure-unit, behavioral
cases run on an in-memory SQLite engine.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from agent.db.queries.cohort import cohort_sql
from agent.db.queries.cohort_breakdown import BreakdownDimension, cohort_breakdown_sql


def _criteria(**overrides):
    """Duck-typed CohortCriteria stand-in — mirrors test_cohort_queries.py's inline criteria."""
    defaults = dict(
        diagnosis_codes=[],
        medication_rxnorm_codes=None,
        condition_text=None,
        age_min=None,
        age_max=None,
        gender=None,
        facility=None,
        last_visit_after=None,
        last_visit_before=None,
        zip_code=None,
        city=None,
        state=None,
        inpatient_admission=False,
        inpatient_admission_date_range=None,
        diagnosis_date_range=None,
        sample_size=20,
    )
    defaults.update(overrides)
    return type("_Criteria", (), defaults)()


# --------------------------------------------------------------------------
# SQL-shape unit tests (no DB needed)
# --------------------------------------------------------------------------


def test_cohort_sql_uses_best_rank_join_not_hard_rank_1():
    sql, _ = cohort_sql(_criteria(gender="F"), schema_prefix="dw.", dialect="postgres")
    assert "empi_rank = 1" not in sql
    assert "ROW_NUMBER() OVER" in sql
    assert "PARTITION BY patient_id" in sql
    assert "COALESCE(empi_rank, 2147483647)" in sql
    assert "source_id ASC" in sql  # deterministic tie-break
    assert "rn = 1" in sql


def test_count_only_path_also_uses_best_rank_join():
    sql, _ = cohort_sql(_criteria(gender="F", sample_size=0), schema_prefix="dw.", dialect="postgres")
    assert "empi_rank = 1" not in sql and "ROW_NUMBER() OVER" in sql


def test_cohort_breakdown_sql_uses_best_rank_join_not_hard_rank_1():
    bucket_sql, total_sql, _ = cohort_breakdown_sql(
        _criteria(gender="F"), BreakdownDimension.GENDER, schema_prefix="dw.", dialect="postgres"
    )
    for sql in (bucket_sql, total_sql):
        assert "empi_rank = 1" not in sql
        assert "ROW_NUMBER() OVER" in sql
        assert "COALESCE(empi_rank, 2147483647)" in sql


# --------------------------------------------------------------------------
# Behavioral tests (real SQLite rows)
# --------------------------------------------------------------------------

# (patient_id, gender, age, full_name, last_date_of_visit, practice_name)
_PATIENTS = [
    (1, "F", 30, "Alice Rank1And2", None, None),
    (2, "F", 40, "Bea RankTwoOnly", None, None),
]

# (patient_id, source_id, empi_rank)
_SOURCE_REFS = [
    (1, "SRC-1-RANK1", 1),
    (1, "SRC-1-RANK2", 2),
    (2, "SRC-2-RANK2-ONLY", 2),  # patient 2 has NO empi_rank=1 row
]


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://")
    with eng.begin() as c:
        c.execute(
            text(
                "CREATE TABLE internal_patient_profile_v ("
                "patient_id INTEGER, gender TEXT, age INTEGER, full_name TEXT, "
                "last_date_of_visit TEXT, practice_name TEXT, city TEXT, state TEXT, "
                "zip_code TEXT)"
            )
        )
        c.execute(
            text("CREATE TABLE internal_source_reference_v (patient_id INTEGER, source_id TEXT, empi_rank INTEGER)")
        )
        for pid, gender, age, name, lv, practice in _PATIENTS:
            c.execute(
                text(
                    "INSERT INTO internal_patient_profile_v "
                    "(patient_id, gender, age, full_name, last_date_of_visit, practice_name) "
                    "VALUES (:pid, :gender, :age, :name, :lv, :practice)"
                ),
                {"pid": pid, "gender": gender, "age": age, "name": name, "lv": lv, "practice": practice},
            )
        for pid, sid, rank in _SOURCE_REFS:
            c.execute(
                text(
                    "INSERT INTO internal_source_reference_v (patient_id, source_id, empi_rank) "
                    "VALUES (:pid, :sid, :rank)"
                ),
                {"pid": pid, "sid": sid, "rank": rank},
            )
    yield eng
    eng.dispose()


def _run(engine, sql, params):
    with engine.connect() as c:
        return c.execute(text(sql), params).mappings().all()


class TestBestRankJoinBehavior:
    def test_patient_without_rank_1_row_is_counted(self, engine):
        """Before the fix, patient 2 (empi_rank=2 only) is invisible; after, both count."""
        sql, params = cohort_sql(_criteria(gender="F", sample_size=5), schema_prefix="", dialect="sqlite")
        rows = _run(engine, sql, params)
        assert len(rows) == 2
        assert rows[0]["total_count"] == 2

    def test_patient_without_rank_1_row_source_id_appears_in_sample(self, engine):
        sql, params = cohort_sql(_criteria(gender="F", sample_size=5), schema_prefix="", dialect="sqlite")
        rows = _run(engine, sql, params)
        source_ids = {r["source_id"] for r in rows}
        assert "SRC-2-RANK2-ONLY" in source_ids

    def test_patient_with_rank_1_row_picks_rank_1_not_rank_2(self, engine):
        """Patient 1 has both rank 1 and rank 2 rows -- the join must pick rank 1."""
        sql, params = cohort_sql(_criteria(gender="F", sample_size=5), schema_prefix="", dialect="sqlite")
        rows = _run(engine, sql, params)
        source_ids = {r["source_id"] for r in rows}
        assert "SRC-1-RANK1" in source_ids
        assert "SRC-1-RANK2" not in source_ids

    def test_count_only_path_counts_patient_without_rank_1_row(self, engine):
        sql, params = cohort_sql(_criteria(gender="F", sample_size=0), schema_prefix="", dialect="sqlite")
        rows = _run(engine, sql, params)
        assert rows[0]["total_count"] == 2
