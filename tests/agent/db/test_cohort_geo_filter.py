"""Tests for the geographic filters added to cohort_sql.

Uses synthetic_db (SQLite + redshift_synthetic.sql) populated with
structured geographic columns on internal_patient_profile_v:
  patient_id=1 (src-john-1962) → zip_code='14223', city='Buffalo', state='NY'
  patient_id=2 (src-john-1971) → zip_code='14201', city='Buffalo', state='NY'
  patient_id=3 (src-jane-1985) → zip_code='15213', city='Pittsburgh', state='PA'

No JOIN to federated_demographic_v — geo filters are direct WHERE clauses
on p.zip_code / p.city / p.state (pivot confirmed by May 2026 warehouse probe).
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
    # Structural: the SQL must filter on p.zip_code directly, not via a JOIN
    assert "p.zip_code = :geo_zip" in sql
    assert "federated_demographic_v" not in sql
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    # patient_id=1 (John 1962, zip 14223) has rank-1, rank-2, and rank-99
    # source references, but the isr join now selects empi_rank = 1 (the
    # current primary CID, one per person), so exactly the rank-1 row
    # (src-john-1962) comes back — NOT the rank-2 src-john-1962-alt.
    assert src_ids == ["src-john-1962"], (
        f"only the rank-1 primary CID for zip-14223 (John 1962) expected; got {src_ids}"
    )
    assert "src-john-1962-alt" not in src_ids, f"rank-2 merged CID must be deduped; got {src_ids}"
    assert "src-jane-1985" not in src_ids, f"Pittsburgh (15213) should not match 14223"
    assert "src-john-1971" not in src_ids, f"zip 14201 should not match 14223"


def test_city_filter_matches_substring(synthetic_db):
    sql, params = cohort_sql(_make(city="Buffalo"), schema_prefix="")
    assert "LOWER(p.city) LIKE :geo_city" in sql
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    # All Buffalo patients (patients 1–2 from synthetic fixture + 4–8 who
    # also have city='Buffalo') — but the isr join uses empi_rank = 1, so
    # each person contributes their single current primary CID. John 1962's
    # rank-2 merged CID (src-john-1962-alt) is deduped away.
    assert "src-john-1962" in src_ids, f"got {src_ids}"
    assert "src-john-1962-alt" not in src_ids, f"rank-2 merged CID must be deduped; got {src_ids}"
    assert "src-john-1971" in src_ids, f"got {src_ids}"
    assert "src-jane-1985" not in src_ids, f"Pittsburgh should not match Buffalo; got {src_ids}"


def test_state_filter_matches_both_2letter_and_full_name():
    """Live regression on 2026-05-13: the warehouse state column carries
    both 'NY' and 'NEW YORK' inconsistently. The tool must accept either
    form as input AND emit SQL that catches both forms in the data,
    otherwise half the population is silently dropped."""
    sql, params = cohort_sql(_make(state="NY"), schema_prefix="dw.")
    forms = sorted(v for k, v in params.items() if k.startswith("geo_state_"))
    assert forms == ["NEW YORK", "NY"], f"expected both alias forms, got {forms}"
    assert "UPPER(p.state) IN" in sql

    # Same when the user passes the full form.
    sql2, params2 = cohort_sql(_make(state="new york"), schema_prefix="dw.")
    forms2 = sorted(v for k, v in params2.items() if k.startswith("geo_state_"))
    assert forms2 == ["NEW YORK", "NY"]


def test_state_filter_unknown_state_falls_back_to_single_form():
    """States not in the alias map should still emit a working SQL clause —
    just the upper-cased input. Don't crash on rare states."""
    sql, params = cohort_sql(_make(state="VT"), schema_prefix="dw.")
    forms = [v for k, v in params.items() if k.startswith("geo_state_")]
    assert forms == ["VT"]
    assert "UPPER(p.state) IN" in sql


def test_state_filter_exact_match_against_synthetic(synthetic_db):
    sql, params = cohort_sql(_make(state="ny"), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    src_ids = sorted(r._mapping["source_id"] for r in rows)
    assert "src-john-1962" in src_ids, f"got {src_ids}"
    assert "src-john-1971" in src_ids, f"got {src_ids}"
    assert "src-jane-1985" not in src_ids, f"PA should not match NY; got {src_ids}"

    # Upper-case "NY" must produce the same result.
    sql2, params2 = cohort_sql(_make(state="NY"), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows2 = conn.execute(text(sql2), params2).fetchall()
    src_ids2 = sorted(r._mapping["source_id"] for r in rows2)
    assert src_ids2 == src_ids, f"'NY' and 'ny' should match identically; got {src_ids2}"


def test_zip_combines_with_diagnosis_codes(synthetic_db):
    """Zip + dx is the failing-question pattern: only patients with both."""
    # Insert a diagnosis row for patient_id=1 (John 1962, zip=14223).
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
    # patient_id=1 (zip 14223) has rank-1/rank-2/rank-99 CIDs; the isr join
    # selects empi_rank = 1, so only the primary CID (src-john-1962) returns.
    assert src_ids == ["src-john-1962"], f"only the rank-1 primary CID for zip-14223 expected; got {src_ids}"
    assert "src-john-1962-alt" not in src_ids, f"rank-2 merged CID must be deduped; got {src_ids}"
    assert "src-john-1971" not in src_ids, f"zip 14201 + no I10 should be excluded; got {src_ids}"


def test_geo_filter_uses_internal_patient_profile_v(synthetic_db):
    """Geo filtering must use p.zip_code on internal_patient_profile_v —
    no JOIN to federated_demographic_v should appear."""
    sql, _ = cohort_sql(_make(zip_code="14223"), schema_prefix="dw.")
    assert "p.zip_code" in sql
    assert "dw.federated_demographic_v" not in sql


def test_sample_size_zero_emits_count_only_fast_path():
    """sample_size=0 must take the count-only fast path. The full pivot
    (SELECT source_id, ..., COUNT(*) OVER ()) forces Postgres to compute
    the window over the full result before LIMIT — burned a 30s timeout
    on broad cohorts. The count-only shape is a pure aggregate that's
    cheap regardless of population size, and counts distinct source_id at
    empi_rank = 1 (each person's current primary CID, one per person) to
    match the sample path's COUNT(*) OVER () grain and the breakdown path."""
    sql, params = cohort_sql(_make(state="NY", sample_size=0), schema_prefix="dw.")
    assert "COUNT(DISTINCT isr.source_id) AS total_count" in sql
    assert "COUNT(*) OVER ()" not in sql
    assert "LIMIT" not in sql.upper()
    # No sample_size_plus_one param — that's only for the per-row path.
    assert "sample_size_plus_one" not in params


def test_sample_size_zero_returns_count_against_synthetic(synthetic_db):
    """End-to-end: sample_size=0 path actually runs and returns a count."""
    sql, params = cohort_sql(_make(state="NY", sample_size=0), schema_prefix="")
    with synthetic_db.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    assert len(rows) == 1
    # Synthetic NY patients are 1,2,4,5,6,7,8 (Jane is PA) = 7 people.
    # The isr join uses empi_rank = 1 (each person's current primary CID,
    # exactly one per person), so the rank-1 count is per-person: John 1962
    # (patient 1) has a rank-1 + a rank-2 + a rank-99 CID but contributes a
    # single rank-1 source_id, same as everyone else → 7 distinct people.
    # (empi_rank != 99 would count John's rank-1 AND rank-2 CIDs → 8, an
    # over-count of the merged person.)
    total = rows[0]._mapping["total_count"]
    assert total == 7, f"expected 7 distinct NY people (rank-1 primary CID), got {total}"


def test_sample_size_nonzero_keeps_per_row_pivot():
    """The default (non-zero) sample_size path must keep the per-row SELECT
    and LIMIT — that's the only way the tool can return sample patient
    rows alongside the total count."""
    sql, params = cohort_sql(_make(state="NY", sample_size=20), schema_prefix="dw.")
    assert "isr.source_id" in sql
    assert "p.full_name" in sql
    assert "COUNT(*) OVER ()" in sql
    assert "LIMIT :sample_size_plus_one" in sql
    assert params["sample_size_plus_one"] == 21
