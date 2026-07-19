"""Fail-closed ConsentGate + authoritative person-grain consent contract (#243/#244).

Covers the union contract: current + history demographic views, claims-key
normalization, EMPI linkage, latest-explicit-by-last_modified_datetime with
FALSE winning ties (revocation-safe), and the NEVER_EXPLICIT bucket.
"""

from sqlalchemy import create_engine, text

from agent.consent.gate import ConsentGate, consent_required
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.consent import consent_population_counts_sql

# (patient_id, source_id, empi_rank) xref rows.
_XREF = [
    (1, "s1", 1),
    (2, "s2", 1),
    (3, "s3", 1),
    (4, "under4", 1),  # claims underlying key
    (5, "s5", 1),  # history-only consent
    (6, "s6", 1),  # no explicit event
    (7, "s7", 99),  # stale rank -> excluded
]
# federated_demographic_v rows: (source_id, source_name, hie_consent, last_modified_datetime)
_CURRENT = [
    ("s1", "BMG", "TRUE", "2026-01-01"),
    ("s2", "BMG", "FALSE", "2026-01-01"),
    ("s3", "BMG", "TRUE", "2026-01-01"),
    ("s3", "BMG", "FALSE", "2026-06-01"),  # revocation: later FALSE wins
    ("pay-mem-under4", "claims_process", "TRUE", "2026-02-01"),  # claims-prefixed
    ("s6", "BMG", "", "2026-01-01"),  # blank only -> never explicit
    ("s7", "BMG", "TRUE", "2026-01-01"),  # but rank 99 -> not linked
]
# federated_demographic_history_v rows.
_HISTORY = [
    ("s5", "BMG", "TRUE", "2025-01-01"),
]


def _insert_demo(conn, table, rows):
    for s, n, cons, d in rows:
        conn.execute(
            text(
                f"INSERT INTO {table} (source_id, source_name, hie_consent, last_modified_datetime) "
                "VALUES (:s, :n, :c, :d)"
            ),
            {"s": s, "n": n, "c": cons, "d": d},
        )


def _adapter():
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as c:
        for t in ("federated_demographic_v", "federated_demographic_history_v"):
            c.execute(
                text(
                    f"CREATE TABLE {t} (source_id TEXT, source_name TEXT, "
                    "hie_consent TEXT, last_modified_datetime TEXT)"
                )
            )
        c.execute(
            text("CREATE TABLE internal_source_reference_v (patient_id INTEGER, source_id TEXT, empi_rank INTEGER)")
        )
        c.execute(text("CREATE TABLE internal_patient_profile_v (patient_id INTEGER)"))
        for pid, sid, rank in _XREF:
            c.execute(
                text("INSERT INTO internal_source_reference_v VALUES (:p, :s, :r)"),
                {"p": pid, "s": sid, "r": rank},
            )
        for pid in {p for p, _, _ in _XREF}:
            c.execute(text("INSERT INTO internal_patient_profile_v VALUES (:p)"), {"p": pid})
        _insert_demo(c, "federated_demographic_v", _CURRENT)
        _insert_demo(c, "federated_demographic_history_v", _HISTORY)
    return AnalyticsDbAdapter(engine=eng, dialect="sqlite")


def test_consented_patient_allowed():
    assert ConsentGate(_adapter()).is_consented(1) is True


def test_explicit_false_denied():
    assert ConsentGate(_adapter()).is_consented(2) is False


def test_revocation_latest_false_wins():
    assert ConsentGate(_adapter()).is_consented(3) is False


def test_claims_normalized_key_links_consent():
    # 'pay-mem-under4' normalizes to 'under4' -> patient 4 -> TRUE.
    assert ConsentGate(_adapter()).is_consented(4) is True


def test_history_view_only_consent_counts():
    assert ConsentGate(_adapter()).is_consented(5) is True


def test_blank_only_denied():
    assert ConsentGate(_adapter()).is_consented(6) is False


def test_stale_rank99_not_linked_denied():
    assert ConsentGate(_adapter()).is_consented(7) is False


def test_unknown_patient_denied():
    assert ConsentGate(_adapter()).is_consented(999) is False


def test_none_patient_denied():
    assert ConsentGate(_adapter()).is_consented(None) is False


def test_non_enforcing_allows_all():
    gate = ConsentGate(_adapter(), enforcing=False)
    assert gate.is_consented(2) is True
    assert gate.is_consented(None) is True


def test_consent_required_default_true_for_all_roles():
    for role in (None, "admin", "doctor", "nurse", "erie_county_clinical"):
        assert consent_required(role) is True


def test_population_counts_true_false_never_explicit():
    adapter = _adapter()
    sql, params = consent_population_counts_sql(dialect="sqlite")
    counts = {r["status"]: r["n"] for r in adapter.fetch_all(sql, params)}
    # Linked-explicit: p1 TRUE, p3 FALSE, p4 TRUE, p5 TRUE => TRUE=3, FALSE=2 (p2,p3).
    assert counts.get("TRUE") == 3
    assert counts.get("FALSE") == 2
    # p6 (blank) and p7 (rank99 only) have no linked explicit event => NEVER_EXPLICIT.
    assert counts.get("NEVER_EXPLICIT") == 2
