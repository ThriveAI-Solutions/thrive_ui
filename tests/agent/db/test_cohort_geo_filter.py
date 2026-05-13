"""Tests for the geographic JOIN added to cohort_sql.

Uses synthetic_db (SQLite + redshift_synthetic.sql) populated with addresses:
  src-john-1962 → '12 Elm St, Buffalo NY 14223'
  src-john-1971 → '88 Oak Ave, Buffalo NY 14201'
  src-jane-1985 → '440 Pine Rd, Pittsburgh PA 15213'

SQLite's ILIKE behavior: SQLite treats LIKE as case-insensitive by default,
which matches Postgres ILIKE semantics for these tests.
"""

from __future__ import annotations
import pytest
from sqlalchemy import text

from agent.db.queries.cohort import cohort_sql


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
    zip_code = None
    city = None
    state = None
    sample_size = 20


def _make(**kw):
    c = _C()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_zip_code_filter_matches_only_that_zip(synthetic_db):
    sql, params = cohort_sql(_make(zip_code="14223"), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    assert src_ids == ["src-john-1962"], f"expected only 14223 match; got {src_ids}"


def test_city_filter_matches_substring(synthetic_db):
    sql, params = cohort_sql(_make(city="Buffalo"), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    assert src_ids == ["src-john-1962", "src-john-1971"], f"got {src_ids}"


def test_state_filter_uses_bracketed_pattern(synthetic_db):
    sql, params = cohort_sql(_make(state="NY"), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    assert src_ids == ["src-john-1962", "src-john-1971"], f"got {src_ids}"


def test_zip_combines_with_diagnosis_codes(synthetic_db):
    """Zip + dx is the failing-question pattern: only patients with both."""
    # First insert a diagnosis row for src-john-1962 so the JOIN has data.
    with synthetic_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO metric_federated_data_v "
                "(patient_id, code, code_type, start_date, is_claims_data) "
                "VALUES (1, 'I10', 'ICD-10', '2025-01-01', 0)"
            )
        )
    sql, params = cohort_sql(
        _make(zip_code="14223", diagnosis_codes=["I10"]),
        schema_prefix="",
    )
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    assert src_ids == ["src-john-1962"], f"got {src_ids}"


def test_schema_prefix_qualifies_geo_join_view(synthetic_db):
    sql, _ = cohort_sql(_make(zip_code="14223"), schema_prefix="dw.")
    assert "dw.federated_demographic_v" in sql


def test_state_filter_matches_trailing_state(synthetic_db):
    """Address ending with state code (no trailing space) must still match."""
    # Insert a row whose address ends with "NY" (no zip after).
    with synthetic_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO federated_demographic_v "
                "(source_id, patient_id, first_name, last_name, date_of_birth, gender, address) "
                "VALUES ('src-trailing-ny', '4', 'Trailing', 'State', '1990-01-01', 'F', '1 Edge Lane, Buffalo NY')"
            )
        )
        # Provide internal_patient_profile_v + internal_source_reference_v
        # rows so the JOIN chain reaches the new demographic row.
        conn.execute(
            text(
                "INSERT INTO internal_patient_profile_v "
                "(patient_id, first_name, last_name, full_name, date_of_birth, age, gender, last_date_of_visit, practice_name, conditions) "
                "VALUES (99, 'Trailing', 'State', 'Trailing State', '1990-01-01', 35, 'F', '2025-01-01', 'BMG', 'none')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO internal_source_reference_v (patient_id, source_id, empi_rank) "
                "VALUES (99, 'src-trailing-ny', 1)"
            )
        )
    sql, params = cohort_sql(_make(state="NY"), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    assert "src-trailing-ny" in src_ids, f"trailing-state address must match; got {src_ids}"
