"""Post-Fusion patient-matching regressions (#239 PID-1, #243 CON-3 matching).

Two identity bugs the warehouse's EMPI shape exposes:

1. ``empi_rank = 1`` is NOT unique per patient — some patients have several
   rank-1 xref rows (find_patient fans them out into duplicate results) and
   some have no rank-1 row at all (find_patient drops them entirely). The fix
   is a deterministic best-rank ``ROW_NUMBER()`` pick, one row per patient.
2. A single ``source_id`` can resolve to more than one internal ``patient_id``.
   The old resolve used ``LIMIT 1`` and silently picked one — which can merge
   or misattribute two different real patients' charts. The fix refuses: >1
   distinct patient_id => treat as not found (fail-closed, same as a
   nonexistent id), logging a count only.
"""

from sqlalchemy import create_engine, text

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.federation import federated_source_ids
from agent.db.queries.patient import find_patient_sql


def _engine():
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as c:
        c.execute(
            text(
                "CREATE TABLE internal_patient_profile_v ("
                "patient_id INTEGER, first_name TEXT, last_name TEXT, full_name TEXT, "
                "date_of_birth TEXT, date_of_death TEXT, age INTEGER, "
                "last_date_of_visit TEXT, practice_name TEXT)"
            )
        )
        c.execute(
            text(
                "CREATE TABLE internal_source_reference_v ("
                "patient_id INTEGER, source_id TEXT, empi_rank INTEGER, "
                "source_name TEXT, source_type TEXT)"
            )
        )
        # patient 10: TWO rank-1 xref rows (fan-out trap)
        # patient 11: only a rank-2 xref row, NO rank-1 (drop trap)
        for pid, fn, ln in ((10, "Al", "Rankone"), (11, "Bo", "Norank")):
            c.execute(
                text(
                    "INSERT INTO internal_patient_profile_v VALUES "
                    "(:pid, :fn, :ln, :full, '1970-01-01', NULL, 55, '2026-01-01', 'Clinic')"
                ),
                {"pid": pid, "fn": fn, "ln": ln, "full": f"{fn} {ln}"},
            )
        for pid, sid, rank in (
            (10, "src-rank1-a", 1),
            (10, "src-rank1-b", 1),
            (10, "src-rank1-stale", 99),
            (11, "src-norank-2", 2),
        ):
            c.execute(
                text("INSERT INTO internal_source_reference_v VALUES (:pid, :sid, :rank, 'SRC', 'EHR')"),
                {"pid": pid, "sid": sid, "rank": rank},
            )
        # An ambiguous source_id shared by two distinct patients (12 and 13).
        for pid in (12, 13):
            c.execute(
                text("INSERT INTO internal_source_reference_v VALUES (:pid, 'src-ambiguous', 1, 'SRC', 'EHR')"),
                {"pid": pid},
            )
    return eng


def test_multiple_rank1_rows_do_not_duplicate_patient():
    adapter = AnalyticsDbAdapter(engine=_engine(), dialect="sqlite")
    sql, params = find_patient_sql(last_name="Rankone", limit=25)
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1, f"patient with two rank-1 rows should appear once, got {len(rows)}"
    # Deterministic tiebreak: lowest source_id wins.
    assert rows[0]["source_id"] == "src-rank1-a"


def test_patient_without_rank1_row_is_still_found():
    adapter = AnalyticsDbAdapter(engine=_engine(), dialect="sqlite")
    sql, params = find_patient_sql(last_name="Norank", limit=25)
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1, "patient whose only xref is rank-2 must not be dropped"
    assert rows[0]["source_id"] == "src-norank-2"


def test_rank_99_still_excluded_from_best_rank_pick():
    adapter = AnalyticsDbAdapter(engine=_engine(), dialect="sqlite")
    sql, params = find_patient_sql(last_name="Rankone", limit=25)
    rows = adapter.fetch_all(sql, params)
    assert all(r["source_id"] != "src-rank1-stale" for r in rows)


def test_ambiguous_source_id_refuses_rather_than_picking_one():
    adapter = AnalyticsDbAdapter(engine=_engine(), dialect="sqlite")
    # A source_id mapping to >1 distinct patient_id must fail closed: no
    # federation set is returned, so downstream retrieval finds nothing —
    # never silently one of the two patients' charts.
    sids = federated_source_ids(adapter, "src-ambiguous")
    assert sids == [], f"ambiguous source_id must refuse, got {sids}"


def test_unambiguous_source_id_federates_normally():
    adapter = AnalyticsDbAdapter(engine=_engine(), dialect="sqlite")
    sids = federated_source_ids(adapter, "src-rank1-a")
    # patient 10's non-stale siblings, deduped; stale rank-99 excluded.
    assert set(sids) == {"src-rank1-a", "src-rank1-b"}
    assert "src-rank1-stale" not in sids
