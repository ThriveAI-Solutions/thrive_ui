"""Patient 360 service wiring (#246)."""

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.patient360.service import run_patient360


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
