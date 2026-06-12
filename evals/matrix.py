"""Question set + roster loading and the question×patient run matrix.

Pure logic — no LLM, no DB. The harness CLI feeds the matrix to the
runner; --dry-run just prints it.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ALLOWED_PARAMS = {"date_start", "date_end", "vaccine", "disease"}


@dataclass
class Question:
    id: str
    title: str
    prompt: str
    followups: list[str]
    reviewer_note: str = ""


@dataclass
class RosterPatient:
    source_id: str
    label: str = ""
    questions: list[str] | None = None  # None = run all
    overrides: dict = field(default_factory=dict)


@dataclass
class PlannedTurn:
    role: str  # "main" | "followup"
    prompt: str


@dataclass
class PlannedConversation:
    conversation_id: str
    question_id: str
    question_title: str
    reviewer_note: str
    source_id: str
    patient_label: str
    turns: list[PlannedTurn]


def load_questions(path: Path) -> list[Question]:
    raw = yaml.safe_load(Path(path).read_text())["questions"]
    questions = [
        Question(
            id=q["id"],
            title=q["title"],
            prompt=q["prompt"],
            followups=list(q.get("followups") or []),
            reviewer_note=q.get("reviewer_note", "") or "",
        )
        for q in raw
    ]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question ids in {path}")
    return questions


def load_roster(path: Path) -> tuple[dict, list[RosterPatient]]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    defaults = {k: str(v) for k, v in (raw.get("defaults") or {}).items()}
    unknown = set(defaults) - ALLOWED_PARAMS
    if unknown:
        raise ValueError(f"Unknown defaults keys in roster: {sorted(unknown)}")
    patients: list[RosterPatient] = []
    for entry in raw.get("patients") or []:
        if not entry.get("source_id"):
            raise ValueError("Every roster patient needs a source_id")
        overrides = {k: str(v) for k, v in entry.items() if k in ALLOWED_PARAMS}
        patients.append(
            RosterPatient(
                source_id=str(entry["source_id"]),
                label=str(entry.get("label", "") or ""),
                questions=[str(q) for q in entry["questions"]] if entry.get("questions") else None,
                overrides=overrides,
            )
        )
    if not patients:
        raise ValueError(f"Roster {path} has no patients")
    return defaults, patients


def _render(template: str, params: dict, question_id: str) -> str:
    needed = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    missing = needed - set(params)
    if missing:
        raise ValueError(
            f"{question_id} needs parameter(s) {sorted(missing)} — add them to roster defaults or the patient entry"
        )
    return template.format(**params)


def build_matrix(
    questions: list[Question],
    defaults: dict,
    patients: list[RosterPatient],
    only: list[str] | None = None,
) -> list[PlannedConversation]:
    by_id = {q.id: q for q in questions}
    matrix: list[PlannedConversation] = []
    for patient in patients:
        wanted = patient.questions or [q.id for q in questions]
        if only:
            wanted = [qid for qid in wanted if qid in only]
        params = {**defaults, **patient.overrides}
        for qid in wanted:
            if qid not in by_id:
                raise ValueError(f"Roster references unknown question id {qid!r}")
            q = by_id[qid]
            turns = [PlannedTurn(role="main", prompt=_render(q.prompt, params, qid))]
            turns += [PlannedTurn(role="followup", prompt=_render(f, params, qid)) for f in q.followups]
            matrix.append(
                PlannedConversation(
                    conversation_id=f"{qid}__{patient.source_id}",
                    question_id=qid,
                    question_title=q.title,
                    reviewer_note=q.reviewer_note,
                    source_id=patient.source_id,
                    patient_label=patient.label,
                    turns=turns,
                )
            )
    return matrix
