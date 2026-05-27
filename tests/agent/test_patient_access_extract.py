# tests/agent/test_patient_access_extract.py
from agent.run_logger import extract_source_ids


def test_extracts_from_matches_list():
    payload = {
        "matches": [
            {"source_id": "src-1", "display_name": "Ann B"},
            {"source_id": "src-2", "display_name": "Cy D"},
        ],
        "total_unique": 2,
    }
    found = extract_source_ids(payload)
    assert ("src-1", "Ann B") in found
    assert ("src-2", "Cy D") in found


def test_extracts_from_nested_items_and_sample():
    payload = {"sample": [{"source_id": "src-9", "display_name": "Zed"}], "items": [{"source_id": "src-7"}]}
    found = dict(extract_source_ids(payload))
    assert found["src-9"] == "Zed"
    assert found["src-7"] is None


def test_dedupes_and_ignores_non_source_id():
    payload = {
        "matches": [{"source_id": "src-1"}, {"source_id": "src-1"}],
        "row_count": 5,
        "data_availability": "data_present",
    }
    found = extract_source_ids(payload)
    assert found == [("src-1", None)]


def test_handles_none_and_non_dict():
    assert extract_source_ids(None) == []
    assert extract_source_ids("nope") == []
    assert extract_source_ids(42) == []
