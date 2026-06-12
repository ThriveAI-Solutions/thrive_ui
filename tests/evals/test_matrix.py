"""Roster parsing, prompt templating, and matrix building."""

from pathlib import Path

import pytest

from evals.matrix import build_matrix, load_questions, load_roster

_REPO = Path(__file__).resolve().parents[2]

_ROSTER = """
defaults:
  date_start: "2024-01-01"
  date_end: "2026-06-11"
  vaccine: "MMR"
  disease: "diabetes"
patients:
  - source_id: "src-a"
    label: "patient A"
  - source_id: "src-b"
    questions: [Q4, Q6]
    vaccine: "Tdap"
"""


@pytest.fixture
def questions():
    return load_questions(_REPO / "evals/questions.yaml")


@pytest.fixture
def roster(tmp_path):
    p = tmp_path / "roster.yaml"
    p.write_text(_ROSTER)
    return load_roster(p)


def test_load_roster_defaults_and_patients(roster):
    defaults, patients = roster
    assert defaults["vaccine"] == "MMR"
    assert [p.source_id for p in patients] == ["src-a", "src-b"]
    assert patients[1].overrides == {"vaccine": "Tdap"}


def test_matrix_full_and_subset(questions, roster):
    defaults, patients = roster
    matrix = build_matrix(questions, defaults, patients)
    # src-a runs all 10 questions, src-b only its 2
    assert len(matrix) == 12
    src_b = [c for c in matrix if c.source_id == "src-b"]
    assert sorted(c.question_id for c in src_b) == ["Q4", "Q6"]


def test_templating_binds_placeholders(questions, roster):
    defaults, patients = roster
    matrix = build_matrix(questions, defaults, patients)
    q6_a = next(c for c in matrix if c.question_id == "Q6" and c.source_id == "src-a")
    q6_b = next(c for c in matrix if c.question_id == "Q6" and c.source_id == "src-b")
    assert "MMR vaccine" in q6_a.turns[0].prompt
    assert "Tdap vaccine" in q6_b.turns[0].prompt
    q1_a = next(c for c in matrix if c.question_id == "Q1" and c.source_id == "src-a")
    assert "between 2024-01-01 and 2026-06-11" in q1_a.turns[0].prompt
    assert q1_a.turns[0].role == "main"
    assert all(t.role == "followup" for t in q1_a.turns[1:])
    assert len(q1_a.turns) == 5  # main + 4 followups


def test_only_filter(questions, roster):
    defaults, patients = roster
    matrix = build_matrix(questions, defaults, patients, only=["Q4"])
    assert {c.question_id for c in matrix} == {"Q4"}


def test_missing_param_raises(questions, tmp_path):
    p = tmp_path / "roster.yaml"
    p.write_text('defaults:\n  date_start: "2024-01-01"\npatients:\n  - source_id: "src-a"\n')
    defaults, patients = load_roster(p)
    with pytest.raises(ValueError, match="date_end"):
        build_matrix(questions, defaults, patients)


def test_unknown_question_id_raises(questions, tmp_path):
    p = tmp_path / "roster.yaml"
    p.write_text(
        'defaults:\n  date_start: "2024-01-01"\n  date_end: "2026-01-01"\n'
        '  vaccine: "MMR"\n  disease: "diabetes"\n'
        'patients:\n  - source_id: "src-a"\n    questions: [Q99]\n'
    )
    defaults, patients = load_roster(p)
    with pytest.raises(ValueError, match="Q99"):
        build_matrix(questions, defaults, patients)
