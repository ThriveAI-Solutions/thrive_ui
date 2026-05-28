"""Tests for agent.db.queries.cohort.cohort_sql.

Builds the SELECT for search_patients_by_criteria. Verifies:
- base SELECT shape (total_count window + columns)
- age band, gender, facility, last_visit, condition_text filters
- diagnosis_codes JOIN on metric_federated_data_v
- medication_rxnorm_codes JOIN on metric_federated_data_v
- LIMIT = sample_size + 1 (truncation sentinel)
- positional placeholders for IN clauses (no SQLAlchemy expanding bindparam)
"""

from __future__ import annotations
from datetime import date
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from agent.db.queries.cohort import cohort_sql


# Minimal CohortCriteria stand-in until Task 3 lands the real model.
# Use a simple namespace with the same attribute names; cohort_sql is
# duck-typed against it.
class _C:
    diagnosis_codes = None
    diagnosis_date_range = None
    medication_rxnorm_codes = None
    condition_text = None
    age_min = None
    age_max = None
    gender = None
    facility = None
    last_visit_after = None
    last_visit_before = None
    sample_size = 20


def _make(**kw):
    c = _C()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_base_query_returns_all_patients(synthetic_db):
    # age_min=0 is a permissive criterion that exercises the base SELECT
    # shape without tripping the empty-criteria guard.
    sql, params = cohort_sql(_make(age_min=0), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    assert len(rows) >= 5, "expected ≥5 patients in base query"
    # total_count window column present and ≥ rows returned
    assert all(r._mapping["total_count"] >= len(rows) for r in rows)


def test_cohort_sql_rejects_empty_criteria():
    """Direct-caller defense: refuse to issue an unfiltered patient scan."""
    with pytest.raises(ValueError):
        cohort_sql(_make())


def test_age_min_filter(synthetic_db):
    sql, params = cohort_sql(_make(age_min=65), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    ages = [r._mapping["age"] for r in rows]
    assert all(a >= 65 for a in ages), f"all ages must be ≥65; got {ages}"


def test_facility_filter_case_insensitive(synthetic_db):
    sql, params = cohort_sql(_make(facility="kaleida"), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    practices = [r._mapping["practice_name"] for r in rows]
    assert all("Kaleida" in p for p in practices), f"all practices must contain 'Kaleida'; got {practices}"


def test_diagnosis_codes_join(synthetic_db):
    sql, params = cohort_sql(_make(diagnosis_codes=["E11.9"]), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    assert len(rows) >= 3, f"expected ≥3 diabetic patients via metric join; got {len(rows)}"


def test_diagnosis_with_facility_and_age(synthetic_db):
    """The acceptance question: diabetic patients over 65 at Kaleida."""
    sql, params = cohort_sql(
        _make(diagnosis_codes=["E11.9"], age_min=65, facility="kaleida"),
        schema_prefix="",
    )
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    assert len(rows) == 2, f"expected exactly 2 (Mary id=4, Susan id=8); got {len(rows)}"


def test_diagnosis_date_range_alone_filters_by_window(synthetic_db):
    """A date window WITHOUT codes counts patients with any ICD-10/SNOMED
    diagnosis whose start_date falls in the window. Nov-Dec 2025 contains
    Daniel (E11.9 2025-11-30) and Susan (E11.9 2025-12-01)."""
    sql, params = cohort_sql(
        _make(diagnosis_date_range=SimpleNamespace(start=date(2025, 11, 1), end=date(2025, 12, 31))),
        schema_prefix="",
    )
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    assert src_ids == ["src-daniel-1977", "src-susan-1955"], f"got {src_ids}"


def test_diagnosis_date_range_alone_excludes_medication_rows(synthetic_db):
    """Jan-Feb 2026 contains only RxNorm rows (metformin); an 'any diagnosis'
    window must NOT count medication events, so the cohort is empty."""
    sql, params = cohort_sql(
        _make(diagnosis_date_range=SimpleNamespace(start=date(2026, 1, 1), end=date(2026, 2, 28))),
        schema_prefix="",
    )
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    assert rows == [], f"expected no diagnosis patients in Jan-Feb 2026; got {[r._mapping['source_id'] for r in rows]}"


def test_diagnosis_date_range_alone_satisfies_criteria_guard(synthetic_db):
    """A bare date window is a real criterion now — cohort_sql must not
    raise the empty-criteria guard."""
    sql, params = cohort_sql(
        _make(diagnosis_date_range=SimpleNamespace(start=date(2025, 1, 1), end=None)),
        schema_prefix="",
    )
    assert "metric_federated_data_v" in sql


def test_medication_rxnorm_codes_join(synthetic_db):
    sql, params = cohort_sql(_make(medication_rxnorm_codes=["6809"]), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    assert len(rows) >= 3, f"expected ≥3 metformin patients; got {len(rows)}"


def test_last_visit_after_excludes_stale(synthetic_db):
    """Anne Garcia (id=6) has last_visit=2024-08-01; filter excludes her."""
    sql, params = cohort_sql(_make(last_visit_after=date(2025, 1, 1)), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = [r._mapping["source_id"] for r in rows]
    assert "src-anne-1948" not in src_ids, f"src-anne-1948 should be filtered out; got {src_ids}"


def test_condition_text_filter(synthetic_db):
    """LIKE on internal_patient_profile_v.conditions."""
    sql, params = cohort_sql(_make(condition_text="hypertension"), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    # Mary (id=4, 'diabetes, hypertension') + Anne (id=6, 'hypertension') match
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    assert src_ids == ["src-anne-1948", "src-mary-1956"], f"got {src_ids}"


def test_sample_size_truncation_sentinel(synthetic_db):
    """LIMIT must be sample_size + 1 so the caller can detect overflow."""
    sql, params = cohort_sql(_make(sample_size=2, age_min=0), schema_prefix="")
    assert ":sample_size_plus_one" in sql
    assert params["sample_size_plus_one"] == 3
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    assert len(rows) == 3, f"expected exactly 3 rows (sample_size=2 + 1 sentinel); got {len(rows)}"


def test_inactive_empi_rank_99_excluded(synthetic_db):
    """John 1962 has an empi_rank=99 row in source_reference; it must not appear."""
    sql, params = cohort_sql(_make(age_min=0), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = [r._mapping["source_id"] for r in rows]
    assert "src-john-1962-stale" not in src_ids, f"empi_rank=99 row leaked; got {src_ids}"


def test_multi_cid_person_counted_once_sample_path(synthetic_db):
    """Dedup proof (sample path): John 1962 (patient_id=1) has a rank-1
    primary CID, a rank-2 legacy/merged CID, and a rank-99 inactive CID.
    The isr join selects empi_rank = 1, so a criterion matching ONLY that
    person (zip 14223) returns exactly his single primary CID — not the
    rank-2 src-john-1962-alt that empi_rank != 99 would have leaked in,
    which over-counted the merged person ~2x."""
    sql, params = cohort_sql(_make(zip_code="14223"), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    assert src_ids == ["src-john-1962"], f"multi-CID person must count once; got {src_ids}"
    assert rows[0]._mapping["total_count"] == 1, (
        f"total_count must be 1 person, not 2 historical CIDs; got {rows[0]._mapping['total_count']}"
    )


def test_multi_cid_person_counted_once_count_only_path(synthetic_db):
    """Dedup proof (count-only fast path): the same single-person criterion
    via sample_size=0 must return total_count == 1, not 2. empi_rank = 1
    selects the one current primary CID per person."""
    sql, params = cohort_sql(_make(zip_code="14223", sample_size=0), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    assert len(rows) == 1
    assert rows[0]._mapping["total_count"] == 1, (
        f"count-only path must count the merged person once; got {rows[0]._mapping['total_count']}"
    )


def test_schema_prefix_applied(synthetic_db):
    """When schema_prefix='dw.', every view reference must be qualified."""
    sql, _ = cohort_sql(_make(diagnosis_codes=["E11.9"]), schema_prefix="dw.")
    assert "dw.internal_patient_profile_v" in sql
    assert "dw.internal_source_reference_v" in sql
    assert "dw.metric_federated_data_v" in sql
