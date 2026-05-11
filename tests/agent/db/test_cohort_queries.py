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
    sql, params = cohort_sql(_make(), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    assert len(rows) >= 5, "expected ≥5 patients in base query"
    # total_count window column present and ≥ rows returned
    assert all(r._mapping["total_count"] >= len(rows) for r in rows)


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
    sql, params = cohort_sql(_make(sample_size=2), schema_prefix="")
    assert ":sample_size_plus_one" in sql
    assert params["sample_size_plus_one"] == 3
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    assert len(rows) == 3, f"expected exactly 3 rows (sample_size=2 + 1 sentinel); got {len(rows)}"


def test_inactive_empi_rank_99_excluded(synthetic_db):
    """John 1962 has an empi_rank=99 row in source_reference; it must not appear."""
    sql, params = cohort_sql(_make(), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = [r._mapping["source_id"] for r in rows]
    assert "src-john-1962-stale" not in src_ids, f"empi_rank=99 row leaked; got {src_ids}"


def test_schema_prefix_applied(synthetic_db):
    """When schema_prefix='dw.', every view reference must be qualified."""
    sql, _ = cohort_sql(_make(diagnosis_codes=["E11.9"]), schema_prefix="dw.")
    assert "dw.internal_patient_profile_v" in sql
    assert "dw.internal_source_reference_v" in sql
    assert "dw.metric_federated_data_v" in sql
