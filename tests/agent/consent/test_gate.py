"""Fail-closed ConsentGate + latest-consent SQL (#243/#244)."""

from sqlalchemy import create_engine, text

from agent.consent.gate import ConsentGate, consent_required
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.consent import consent_population_counts_sql


def _adapter(rows):
    """rows: list of (source_id, hie_consent, created_date)."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as c:
        c.execute(
            text(
                "CREATE TABLE federated_demographic_v ("
                "source_id TEXT, patient_id TEXT, hie_consent TEXT, created_date TEXT)"
            )
        )
        for sid, consent, created in rows:
            c.execute(
                text(
                    "INSERT INTO federated_demographic_v "
                    "(source_id, patient_id, hie_consent, created_date) "
                    "VALUES (:sid, '1', :consent, :created)"
                ),
                {"sid": sid, "consent": consent, "created": created},
            )
    return AnalyticsDbAdapter(engine=eng, dialect="sqlite")


def test_consented_patient_allowed():
    gate = ConsentGate(_adapter([("s1", "TRUE", "2026-01-01")]))
    assert gate.is_consented(["s1"]) is True


def test_explicit_false_denied():
    gate = ConsentGate(_adapter([("s1", "FALSE", "2026-01-01")]))
    assert gate.is_consented(["s1"]) is False


def test_blank_only_denied():
    gate = ConsentGate(_adapter([("s1", "", "2026-01-01"), ("s1", "   ", "2026-02-01")]))
    assert gate.is_consented(["s1"]) is False


def test_latest_explicit_wins_revocation():
    # TRUE then a later FALSE => revoked => denied (latest explicit wins).
    gate = ConsentGate(_adapter([("s1", "TRUE", "2026-01-01"), ("s1", "FALSE", "2026-06-01")]))
    assert gate.is_consented(["s1"]) is False


def test_blank_after_true_keeps_true():
    # A later blank is unknown, not a state; the latest EXPLICIT value (TRUE) holds.
    gate = ConsentGate(_adapter([("s1", "TRUE", "2026-01-01"), ("s1", "", "2026-06-01")]))
    assert gate.is_consented(["s1"]) is True


def test_consent_read_across_federated_siblings():
    # Consent on any sibling source_id counts; latest explicit across the set.
    gate = ConsentGate(_adapter([("s1", "", "2026-01-01"), ("s2", "TRUE", "2026-05-01")]))
    assert gate.is_consented(["s1", "s2"]) is True


def test_empty_source_ids_denied():
    gate = ConsentGate(_adapter([("s1", "TRUE", "2026-01-01")]))
    assert gate.is_consented([]) is False


def test_no_row_denied():
    gate = ConsentGate(_adapter([("s1", "TRUE", "2026-01-01")]))
    assert gate.is_consented(["unknown-sid"]) is False


def test_non_enforcing_allows_all():
    gate = ConsentGate(_adapter([("s1", "FALSE", "2026-01-01")]), enforcing=False)
    assert gate.is_consented(["s1"]) is True
    assert gate.is_consented([]) is True


def test_consent_required_default_true_for_all_roles():
    for role in (None, "admin", "doctor", "nurse", "erie_county_clinical"):
        assert consent_required(role) is True


def test_population_counts_bucket_true_false_blank():
    adapter = _adapter(
        [
            ("s1", "TRUE", "2026-01-01"),
            ("s2", "FALSE", "2026-01-01"),
            ("s3", "", "2026-01-01"),
            ("s4", "  ", "2026-01-01"),
        ]
    )
    sql, params = consent_population_counts_sql()
    row = adapter.fetch_all(sql, params)[0]
    assert row["consent_true"] == 1
    assert row["consent_false"] == 1
    assert row["consent_blank"] == 2
    assert row["total_rows"] == 4
