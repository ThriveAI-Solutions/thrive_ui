from evals.discovery import DOMAIN_PROBES, format_roster_snippet, suggest_patients


def test_suggests_patients_with_data(synthetic_adapter):
    out = suggest_patients(synthetic_adapter, "2026-01-01", "2026-12-31", per_domain=5)
    assert "encounters" in out
    enc_ids = [r["source_id"] for r in out["encounters"]]
    assert "src-john-1962" in enc_ids
    assert all(r["record_count"] >= 1 for r in out["encounters"])
    assert set(out) == {p["domain"] for p in DOMAIN_PROBES}


def test_snippet_merges_domains_per_patient(synthetic_adapter):
    out = suggest_patients(synthetic_adapter, "2026-01-01", "2026-12-31")
    snippet = format_roster_snippet(out)
    assert 'source_id: "src-john-1962"' in snippet
    assert snippet.count('source_id: "src-john-1962"') == 1  # merged, not repeated
    assert "Q5" in snippet  # encounters → Q5 hint


def test_end_date_is_inclusive(synthetic_adapter):
    # enc-1 for src-john-1962 is at 2026-04-01 09:30 — after midnight on the end date
    out = suggest_patients(synthetic_adapter, "2026-04-01", "2026-04-01")
    assert "src-john-1962" in [r["source_id"] for r in out["encounters"]]
