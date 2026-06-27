import json

from evals.report import generate_html


def _fixture_results():
    return {
        "run_id": "eval-test-1",
        "created_at": "2026-06-11T21:00:00",
        "model": {"provider": "ollama", "model": "qwen3.6:27b"},
        "defaults": {"date_start": "2024-01-01", "date_end": "2026-06-11"},
        "conversations": [
            {
                "conversation_id": "Q4__src-a",
                "question_id": "Q4",
                "question_title": "Diabetes history and latest A1C",
                "reviewer_note": "",
                "patient": {"source_id": "src-a", "display_name": "John Smith", "label": "demo"},
                "status": "ok",
                "error": None,
                "turns": [
                    {
                        "index": 0,
                        "role": "main",
                        "prompt": "Does this patient have a history of diabetes?",
                        "answer": "Yes </script> A1C 7.2.",
                        "thinking": ["checking labs"],
                        "tool_calls": [
                            {
                                "tool_name": "get_patient_clinical_data",
                                "arguments": {},
                                "completed": True,
                                "result_summary": "2 rows",
                                "success": True,
                                "elapsed_ms": 2300,
                                "error": None,
                                "sql_executed": [{"sql": "SELECT 1", "params": {}, "db_elapsed_ms": 2100}],
                            }
                        ],
                        "cap_reached": None,
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                        "total_elapsed_ms": 9000,
                        "latency": {"total_ms": 9000, "llm_ms": 6700, "tool_overhead_ms": 200, "redshift_ms": 2100},
                        "judge": {"suggestion": "looks_correct", "reason": "Matches lab rows."},
                    }
                ],
            }
        ],
    }


def test_html_is_standalone():
    html = generate_html(_fixture_results())
    assert html.lstrip().lower().startswith("<!doctype html>")
    # "Standalone" means the page loads NO external resources — not that the
    # string "http://" never appears. The vendored echarts.min.js legitimately
    # contains W3C XML-namespace URIs (http://www.w3.org/2000/svg, xlink, …)
    # and Apache/MIT license URLs in comments; those are inert identifiers, not
    # fetches. Assert on the actual external-load vectors instead.
    for vector in ('src="http', "src='http", 'href="http', "href='http", "url(http", 'src="//', 'href="//'):
        assert vector not in html, f"report loads an external resource via {vector!r}"


def test_html_embeds_data_and_marking_machinery():
    html = generate_html(_fixture_results())
    assert 'id="eval-data"' in html
    assert "localStorage" in html
    assert "eval-test-1" in html
    assert "exportVerdicts" in html


def test_embedded_json_escapes_close_tags():
    html = generate_html(_fixture_results())
    assert "<\\/script>" in html  # the answer's </script> can't break the data block


def test_embedded_json_round_trips():
    html = generate_html(_fixture_results())
    start = html.index('id="eval-data">') + len('id="eval-data">')
    end = html.index("</script>", start)
    data = json.loads(html[start:end])
    assert data["conversations"][0]["turns"][0]["latency"]["redshift_ms"] == 2100
