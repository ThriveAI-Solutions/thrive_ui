"""find_patient consent enforcement (#244).

With enforcement on, a non-consented patient is omitted from results and is
indistinguishable from a nonexistent one. With enforcement off (default), all
patients return — preserving today's behavior until consent data + policy land.
"""

from unittest.mock import MagicMock

from sqlalchemy import create_engine, text

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.deps import AgentDeps
from agent.tools.find_patient import PatientSearchQuery, find_patient


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
        for t in ("federated_demographic_v", "federated_demographic_history_v"):
            c.execute(
                text(
                    f"CREATE TABLE {t} (source_id TEXT, source_name TEXT, "
                    "hie_consent TEXT, last_modified_datetime TEXT)"
                )
            )
        # Two patients named Consenttest: 20 consented, 21 not.
        for pid, fn in ((20, "Yes"), (21, "No")):
            c.execute(
                text(
                    "INSERT INTO internal_patient_profile_v VALUES "
                    "(:pid, :fn, 'Consenttest', :full, '1970-01-01', NULL, 55, '2026-01-01', 'Clinic')"
                ),
                {"pid": pid, "fn": fn, "full": f"{fn} Consenttest"},
            )
        for pid, sid in ((20, "src-consented"), (21, "src-declined")):
            c.execute(
                text("INSERT INTO internal_source_reference_v VALUES (:pid, :sid, 1, 'SRC', 'EHR')"),
                {"pid": pid, "sid": sid},
            )
        for sid, consent in (("src-consented", "TRUE"), ("src-declined", "FALSE")):
            c.execute(
                text(
                    "INSERT INTO federated_demographic_v "
                    "(source_id, source_name, hie_consent, last_modified_datetime) "
                    "VALUES (:sid, 'SRC', :consent, '2026-01-01')"
                ),
                {"sid": sid, "consent": consent},
            )
    return eng


def _deps(engine, *, enforce_consent):
    return AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=None,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=AnalyticsDbAdapter(engine=engine, dialect="sqlite"),
        rag=None,
        sqlite_session=None,
        run_logger=MagicMock(),
        enforce_consent=enforce_consent,
    )


def test_enforcement_off_returns_both():
    ctx = MagicMock()
    ctx.deps = _deps(_engine(), enforce_consent=False)
    result = find_patient(ctx, PatientSearchQuery(last_name="Consenttest"))
    assert sorted(m.source_id for m in result.matches) == ["src-consented", "src-declined"]


def test_enforcement_on_omits_non_consented():
    ctx = MagicMock()
    ctx.deps = _deps(_engine(), enforce_consent=True)
    result = find_patient(ctx, PatientSearchQuery(last_name="Consenttest"))
    assert [m.source_id for m in result.matches] == ["src-consented"]
    assert result.total_unique == 1
