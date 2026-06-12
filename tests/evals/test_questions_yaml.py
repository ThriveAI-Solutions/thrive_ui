"""Lint the committed question set: ids, placeholders, shape."""

import string
from pathlib import Path

import yaml

_QUESTIONS = Path(__file__).resolve().parents[2] / "evals/questions.yaml"
_ALLOWED_PARAMS = {"date_start", "date_end", "vaccine", "disease"}


def _placeholders(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def _load():
    return yaml.safe_load(_QUESTIONS.read_text())["questions"]


def test_ten_questions_with_unique_ids():
    questions = _load()
    ids = [q["id"] for q in questions]
    assert ids == [f"Q{i}" for i in range(1, 11)]


def test_required_fields_present():
    for q in _load():
        assert q.get("title"), q["id"]
        assert q.get("prompt"), q["id"]
        assert isinstance(q.get("followups"), list), q["id"]


def test_only_known_placeholders():
    for q in _load():
        used = _placeholders(q["prompt"])
        for f in q["followups"]:
            used |= _placeholders(f)
        assert used <= _ALLOWED_PARAMS, f"{q['id']} uses unknown placeholders {used - _ALLOWED_PARAMS}"


def test_q10_is_decline_check():
    q10 = [q for q in _load() if q["id"] == "Q10"][0]
    assert q10["followups"] == []
    note = q10.get("reviewer_note")
    assert isinstance(note, str) and note.strip(), "Q10 reviewer_note must be a non-empty string"
