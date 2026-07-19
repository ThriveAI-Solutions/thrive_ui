"""Patient 360 fingerprint cache (#246): pipeline seam + SQLite backend."""

from sqlalchemy import create_engine

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.patient360.cache import SqlitePatient360Cache
from agent.patient360.pipeline import generate_patient360

_PATIENT = "src-john-1962"


class DictCache:
    """In-memory SectionCache: stores (fingerprint, narrative) per (source_id, section)."""

    def __init__(self):
        self.store: dict = {}
        self.gets = 0
        self.puts = 0

    def get(self, source_id, section, fingerprint):
        self.gets += 1
        v = self.store.get((source_id, section))
        return v[1] if v and v[0] == fingerprint else None

    def put(self, source_id, section, fingerprint, narrative):
        self.puts += 1
        self.store[(source_id, section)] = (fingerprint, narrative)


class CountingSummarizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, system_prompt, table):
        self.calls += 1
        return f"n{self.calls}"


def test_second_run_hits_cache_and_skips_summarizer(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    cache = DictCache()

    first = CountingSummarizer()
    r1 = generate_patient360(adapter, _PATIENT, summarizer=first, cache=cache)
    done1 = [s for s in r1.sections if s.status == "done"]
    assert first.calls == len(done1) + 1  # sections + synthesis; nothing cached yet
    assert all(not s.cached for s in r1.sections)

    second = CountingSummarizer()
    r2 = generate_patient360(adapter, _PATIENT, summarizer=second, cache=cache)
    done2 = [s for s in r2.sections if s.status == "done"]
    # Every section is a cache hit now; only the synthesis pass calls the model.
    assert all(s.cached for s in done2)
    assert second.calls == 1


def test_stale_fingerprint_regenerates(synthetic_db):
    # A changed input yields a different fingerprint; the pipeline must treat a
    # non-matching stored fingerprint as a miss and regenerate. (The synthetic
    # warehouse is read-only, so we simulate the data change by staling the
    # stored fingerprint — the exact condition a real data change produces.)
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    cache = DictCache()
    generate_patient360(adapter, _PATIENT, summarizer=CountingSummarizer(), cache=cache)

    key = next(k for k in cache.store if k[1] == "diagnoses")
    _, narrative = cache.store[key]
    cache.store[key] = ("STALE-FINGERPRINT", narrative)

    r = generate_patient360(adapter, _PATIENT, summarizer=CountingSummarizer(), cache=cache)
    diagnoses = next(s for s in r.sections if s.name == "diagnoses")
    assert diagnoses.cached is False  # stale fingerprint → regenerated
    assert cache.store[key][0] != "STALE-FINGERPRINT"  # cache refreshed to the real fingerprint


def test_sqlite_cache_roundtrip_and_fingerprint_match():
    eng = create_engine("sqlite:///:memory:")
    from orm.models import Base

    Base.metadata.create_all(eng, tables=[Base.metadata.tables["thrive_patient360_section"]])
    from sqlalchemy.orm import Session

    with Session(eng) as s:
        cache = SqlitePatient360Cache(s)
        assert cache.get("p1", "labs", "fp1") is None  # miss
        cache.put("p1", "labs", "fp1", "narrative-A")
        assert cache.get("p1", "labs", "fp1") == "narrative-A"  # hit
        assert cache.get("p1", "labs", "fp2") is None  # fingerprint mismatch => miss
        cache.put("p1", "labs", "fp2", "narrative-B")  # upsert same (source,section)
        assert cache.get("p1", "labs", "fp2") == "narrative-B"
