"""Patient 360 service wiring (#246) + consent gating (#244)."""

from sqlalchemy import create_engine, text

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.patient360.service import run_patient360


def _consent_adapter(consent):
    """Minimal warehouse with one patient (id 50, src-p50) whose latest explicit
    consent is `consent` ('TRUE'/'FALSE')."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as c:
        c.execute(
            text("CREATE TABLE internal_source_reference_v (patient_id INTEGER, source_id TEXT, empi_rank INTEGER)")
        )
        c.execute(text("INSERT INTO internal_source_reference_v VALUES (50, 'src-p50', 1)"))
        for t in ("federated_demographic_v", "federated_demographic_history_v"):
            c.execute(
                text(
                    f"CREATE TABLE {t} (source_id TEXT, source_name TEXT, hie_consent TEXT, last_modified_datetime TEXT)"
                )
            )
        c.execute(
            text(
                "INSERT INTO federated_demographic_v (source_id, source_name, hie_consent, last_modified_datetime) "
                "VALUES ('src-p50', 'SRC', :consent, '2026-01-01')"
            ),
            {"consent": consent},
        )
    return AnalyticsDbAdapter(engine=eng, dialect="sqlite")


def test_non_consented_patient_fails_closed_to_empty_without_summarizing():
    calls = []

    def fake_summarizer(system_prompt, table):
        calls.append(system_prompt)
        return "summary"

    result = run_patient360(
        "src-p50", adapter=_consent_adapter("FALSE"), summarizer=fake_summarizer, enforce_consent=True
    )
    assert result.sections == []
    assert result.master_summary is None
    assert calls == []  # denied before any domain was summarized


def test_enforcement_off_ignores_consent(synthetic_db):
    # enforce_consent defaults off — a patient with no consent data still runs.
    result = run_patient360(
        "src-john-1962", adapter=AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite"), summarizer=lambda sp, t: "s"
    )
    assert any(s.status == "done" for s in result.sections)


def test_run_patient360_with_injected_adapter_and_summarizer(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    calls = []

    def fake_summarizer(system_prompt, table):
        calls.append(system_prompt)
        return "summary"

    result = run_patient360("src-john-1962", adapter=adapter, summarizer=fake_summarizer)

    assert result.source_id == "src-john-1962"
    assert any(s.status == "done" for s in result.sections)
    assert result.master_summary == "summary"
    assert calls  # summarizer was invoked
