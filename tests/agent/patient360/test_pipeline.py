"""Patient 360 deterministic pipeline (#246).

Runs the real curated builders against the synthetic fixture with a fake
summarizer, so the assembly — section order, the visits union, empty/failed
handling, and synthesis input — is verified without a live model.
"""

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.patient360.pipeline import generate_patient360
from agent.patient360.prompts import SECTION_ORDER, SECTION_PROMPTS, SYNTHESIS_SYSTEM_PROMPT

_PATIENT = "src-john-1962"


class FakeSummarizer:
    def __init__(self, fail_on_prompt: str | None = None):
        self.calls: list[tuple[str, str]] = []
        self._fail_on = fail_on_prompt

    def __call__(self, system_prompt: str, table: str) -> str:
        self.calls.append((system_prompt, table))
        if self._fail_on and self._fail_on in system_prompt:
            raise RuntimeError("summarizer boom")
        return f"NARRATIVE[{len(table)}]"


def _adapter(synthetic_db):
    return AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")


def test_runs_every_section_in_fixed_order(synthetic_db):
    result = generate_patient360(_adapter(synthetic_db), _PATIENT, summarizer=FakeSummarizer())
    assert [s.name for s in result.sections] == SECTION_ORDER


def test_visits_unions_encounters_and_admissions(synthetic_db):
    result = generate_patient360(_adapter(synthetic_db), _PATIENT, summarizer=FakeSummarizer())
    by = {s.name: s for s in result.sections}
    assert by["visits"].row_count == by["encounters"].row_count + by["admissions"].row_count
    assert by["encounters"].row_count > 0 and by["admissions"].row_count > 0


def test_visits_markdown_carries_both_feeds_fields(synthetic_db):
    """The visits table mixes EncounterItem and AdmissionStay rows whose fields
    barely overlap; the rendered markdown must carry BOTH feeds' columns and
    values, not render one feed as blank rows."""
    fake = FakeSummarizer()
    generate_patient360(_adapter(synthetic_db), _PATIENT, summarizer=fake, sections=["visits"])
    visits_call = next(c for c in fake.calls if c[0] == SECTION_PROMPTS["visits"])
    table = visits_call[1]
    # encounter-feed column and admission-feed column both present
    assert "event_datetime" in table
    assert "admit_date" in table
    # admission rows carry real values, not blanks: the fixture's inpatient
    # stay dates appear in the table
    assert "2025-06-15" in table


def test_summarizer_called_once_per_done_section_plus_synthesis(synthetic_db):
    fake = FakeSummarizer()
    result = generate_patient360(_adapter(synthetic_db), _PATIENT, summarizer=fake)
    done = [s for s in result.sections if s.status == "done"]
    assert done, "fixture patient should have some populated domains"
    assert len(fake.calls) == len(done) + 1  # +1 synthesis pass


def test_empty_sections_are_marked_and_not_summarized(synthetic_db):
    fake = FakeSummarizer()
    result = generate_patient360(_adapter(synthetic_db), _PATIENT, summarizer=fake)
    called_prompts = {c[0] for c in fake.calls}
    for s in result.sections:
        if s.status == "empty":
            assert s.narrative is None and s.row_count == 0
            assert SECTION_PROMPTS[s.name] not in called_prompts


def test_master_summary_synthesized_from_narratives(synthetic_db):
    fake = FakeSummarizer()
    result = generate_patient360(_adapter(synthetic_db), _PATIENT, summarizer=fake)
    assert result.master_summary is not None
    synth_call = [c for c in fake.calls if c[0] == SYNTHESIS_SYSTEM_PROMPT]
    assert len(synth_call) == 1
    # synthesis input carries the per-section narratives
    assert "### demographics" in synth_call[0][1] or "### diagnoses" in synth_call[0][1]


def test_failed_section_recorded_without_aborting(synthetic_db):
    # Summarizer raises only on the demographics prompt.
    fake = FakeSummarizer(fail_on_prompt=SECTION_PROMPTS["demographics"])
    result = generate_patient360(_adapter(synthetic_db), _PATIENT, summarizer=fake)
    by = {s.name: s for s in result.sections}
    assert by["demographics"].status == "failed"
    assert by["demographics"].error is not None
    # other populated domains still summarized; run still produced a master summary
    assert any(s.status == "done" for s in result.sections)
    assert result.master_summary is not None
