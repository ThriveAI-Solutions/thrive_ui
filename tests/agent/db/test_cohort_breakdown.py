"""Tests for agent.db.queries.cohort_breakdown."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from agent.db.queries.cohort_breakdown import (
    BreakdownDimension,
    breakdown_bucket,
    cohort_breakdown_sql,
)


def test_gender_bucket_is_additive_no_anchor():
    spec = breakdown_bucket(BreakdownDimension.GENDER, dialect="sqlite")
    assert spec.non_additive is False
    assert spec.requires_diagnosis_anchor is False
    assert "p.gender" in spec.group_expr
    assert "Unknown" in spec.group_expr  # null/U folded to a bucket


def test_age_band_bucket_is_additive_no_anchor():
    spec = breakdown_bucket(BreakdownDimension.AGE_BAND, dialect="sqlite")
    assert spec.non_additive is False
    assert spec.requires_diagnosis_anchor is False
    assert "p.age" in spec.group_expr
    assert "65" in spec.group_expr


def test_diagnosis_month_is_non_additive_needs_anchor_sqlite():
    spec = breakdown_bucket(BreakdownDimension.DIAGNOSIS_MONTH, dialect="sqlite")
    assert spec.non_additive is True
    assert spec.requires_diagnosis_anchor is True
    assert "strftime" in spec.group_expr
    assert "dx.start_date" in spec.group_expr


def test_diagnosis_month_uses_date_trunc_on_postgres():
    spec = breakdown_bucket(BreakdownDimension.DIAGNOSIS_MONTH, dialect="postgres")
    assert "DATE_TRUNC" in spec.group_expr or "TO_CHAR" in spec.group_expr
    assert "dx.start_date" in spec.group_expr


def test_unknown_dialect_rejected():
    with pytest.raises(ValueError):
        breakdown_bucket(BreakdownDimension.DIAGNOSIS_YEAR, dialect="oracle")


def _crit(**kw):
    base = dict(
        diagnosis_codes=None,
        diagnosis_date_range=None,
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
        inpatient_admission=None,
        inpatient_admission_date_range=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_gender_breakdown_buckets_sum_to_total(synthetic_db):
    crit = _crit(age_min=0)  # permissive: whole population
    bucket_sql, total_sql, params = cohort_breakdown_sql(
        crit, BreakdownDimension.GENDER, schema_prefix="", dialect="sqlite"
    )
    with synthetic_db.connect() as conn:
        buckets = conn.execute(text(bucket_sql), params).fetchall()
        total = conn.execute(text(total_sql), params).scalar()
    by_count = {r._mapping["bucket_label"]: r._mapping["patient_count"] for r in buckets}
    assert sum(by_count.values()) == total, "single-valued gender must be additive"
    # Per-person grain: the isr join uses empi_rank = 1 (one current primary
    # CID per person), so the 8 synthetic patients count as 8 people total —
    # John 1962's rank-2 merged CID is deduped. (empi_rank != 99 would count
    # his rank-1 + rank-2 CIDs as 9, over-counting the M bucket.)
    assert total == 8, f"expected 8 distinct people (rank-1 primary CID), got {total}"
    assert by_count.get("M") == 4, f"M = patients 1,2,5,7 counted once each; got {by_count.get('M')}"
    assert by_count.get("F") == 4, f"F = patients 3,4,6,8; got {by_count.get('F')}"


def test_diagnosis_month_breakdown_groups_by_month(synthetic_db):
    crit = _crit(diagnosis_date_range=SimpleNamespace(start=date(2025, 1, 1), end=date(2025, 12, 31)))
    bucket_sql, total_sql, params = cohort_breakdown_sql(
        crit, BreakdownDimension.DIAGNOSIS_MONTH, schema_prefix="", dialect="sqlite"
    )
    with synthetic_db.connect() as conn:
        buckets = conn.execute(text(bucket_sql), params).fetchall()
    labels = {r._mapping["bucket_label"] for r in buckets}
    # synthetic ICD-10 diagnoses in 2025: 2025-04, 2025-06, 2025-09, 2025-11, 2025-12
    assert "2025-06" in labels
    assert "2025-09" in labels
    assert all(lbl.startswith("2025-") for lbl in labels)


def test_diagnosis_month_breakdown_dedups_multi_event(synthetic_db):
    """Within-bucket multi-event dedup: Susan (patient 8) has TWO E11.9 rows
    in 2025-12 (enc-8 + enc-8b). The month join fans her out to two rows in
    that bucket, but COUNT(DISTINCT source_id) must collapse them to one
    person — not double-count a patient who saw a provider twice in a month.
    This is the most likely way an event-join could reintroduce a duplicate."""
    crit = _crit(
        diagnosis_codes=["E11.9"],
        diagnosis_date_range=SimpleNamespace(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )
    bucket_sql, total_sql, params = cohort_breakdown_sql(
        crit, BreakdownDimension.DIAGNOSIS_MONTH, schema_prefix="", dialect="sqlite"
    )
    with synthetic_db.connect() as conn:
        buckets = conn.execute(text(bucket_sql), params).fetchall()
    by_count = {r._mapping["bucket_label"]: r._mapping["patient_count"] for r in buckets}
    assert by_count.get("2025-12") == 1, (
        f"Susan's two E11.9 events in 2025-12 must count as one person; got {by_count.get('2025-12')}"
    )


def test_time_breakdown_without_anchor_raises():
    crit = _crit(gender="F")  # demographic only, no diagnosis anchor
    with pytest.raises(ValueError, match="diagnosis"):
        cohort_breakdown_sql(crit, BreakdownDimension.DIAGNOSIS_MONTH, schema_prefix="", dialect="sqlite")


def test_admission_month_bucket_needs_admission_anchor():
    spec = breakdown_bucket(BreakdownDimension.ADMISSION_MONTH, dialect="sqlite")
    assert spec.non_additive is True
    assert spec.requires_inpatient_admission_anchor is True
    assert "qualifying_admit_date" in spec.group_expr
    assert "strftime" in spec.group_expr


def test_admission_month_breakdown_groups_by_month(synthetic_db):
    crit = _crit(inpatient_admission=True)
    bucket_sql, total_sql, params = cohort_breakdown_sql(
        crit, BreakdownDimension.ADMISSION_MONTH, schema_prefix="", dialect="sqlite"
    )
    # exactly one adt_ip join (the filter-only join must be suppressed when the
    # projected admission-date join is present).
    assert bucket_sql.count(") adt_ip ON") == 1
    assert "adt_ip.patient_id = CAST(p.patient_id AS VARCHAR)" in bucket_sql
    with synthetic_db.connect() as conn:
        buckets = conn.execute(text(bucket_sql), params).fetchall()
    by_count = {r._mapping["bucket_label"]: r._mapping["patient_count"] for r in buckets}
    # qualifying inpatient admit dates: p4/V401 2024-01, p1/V100 2025-06,
    # p2/V200 + p4/V400 both 2026-03.
    assert by_count.get("2024-01") == 1
    assert by_count.get("2025-06") == 1
    assert by_count.get("2026-03") == 2


def test_admission_breakdown_without_anchor_raises():
    crit = _crit(gender="F")  # no inpatient_admission anchor
    with pytest.raises(ValueError, match="inpatient_admission"):
        cohort_breakdown_sql(crit, BreakdownDimension.ADMISSION_MONTH, schema_prefix="", dialect="sqlite")
