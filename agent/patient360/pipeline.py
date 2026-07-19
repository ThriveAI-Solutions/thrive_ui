"""Deterministic Patient 360 pipeline (#246).

Fetch each clinical domain via the existing curated builders (federating the
selected patient's source_ids once), summarize each on its own, then synthesize.
The ``visits`` section unions the encounter and ADT feeds (#247 Q7/Q10).

The LLM is injected as ``summarizer(system_prompt, table_markdown) -> str`` so
the assembly is testable without a live model.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, List, Optional, Protocol

from pydantic import BaseModel

from agent.db.federation import federated_source_ids
from agent.patient360.prompts import (
    GENERATOR_VERSION,
    SECTION_ORDER,
    SECTION_PROMPTS,
    SYNTHESIS_SYSTEM_PROMPT,
)
from agent.tools.get_patient_clinical_data import (
    AdmissionsQuery,
    AllergiesQuery,
    DiagnosesQuery,
    EncountersQuery,
    ImagingQuery,
    ImmunizationsQuery,
    LabsQuery,
    MedicationsQuery,
    ProceduresQuery,
    SurgeriesQuery,
    _build_admissions_result,
    _build_allergies_result,
    _build_demographics_result,
    _build_diagnoses_result,
    _build_encounters_result,
    _build_imaging_result,
    _build_immunizations_result,
    _build_labs_result,
    _build_medications_result,
    _build_procedures_result,
    _build_surgeries_result,
)

Summarizer = Callable[[str, str], str]

_ROW_CAP = 200


class SectionCache(Protocol):
    """Persistence seam for section summaries (fingerprint-invalidated).

    ``get`` returns the stored narrative for (source_id, section) ONLY if the
    stored fingerprint equals ``fingerprint`` (same input, same generator);
    otherwise None. ``put`` stores/overwrites.
    """

    def get(self, source_id: str, section: str, fingerprint: str) -> Optional[str]: ...

    def put(self, source_id: str, section: str, fingerprint: str, narrative: str) -> None: ...


def _fingerprint(section_markdown: str) -> str:
    """Content fingerprint of a section's summarized input, versioned so a
    GENERATOR_VERSION bump invalidates every stored summary."""
    digest = hashlib.sha256(section_markdown.encode("utf-8")).hexdigest()
    return f"v{GENERATOR_VERSION}:{digest}"


class Patient360Section(BaseModel):
    name: str
    status: str  # "done" | "empty" | "failed"
    row_count: int = 0
    narrative: Optional[str] = None
    cached: bool = False
    error: Optional[str] = None


class Patient360Result(BaseModel):
    source_id: str
    generator_version: int = GENERATOR_VERSION
    sections: List[Patient360Section]
    master_summary: Optional[str] = None


def _items_to_markdown(items: list[Any]) -> str:
    """Compact markdown table of item rows (item_type column dropped).

    Columns are the first-appearance-ordered union across ALL rows — the
    ``visits`` section mixes EncounterItem and AdmissionStay rows whose fields
    barely overlap, and first-row-only columns would render the other feed's
    rows as blanks.
    """
    dumps = [i.model_dump(mode="json") if isinstance(i, BaseModel) else dict(i) for i in items[:_ROW_CAP]]
    columns = [c for c in dict.fromkeys(c for d in dumps for c in d.keys()) if c != "item_type"]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(d.get(c, "") if d.get(c) is not None else "") for c in columns) + " |" for d in dumps
    )
    note = f"\n\n(showing {len(dumps)} of {len(items)} rows)" if len(items) > _ROW_CAP else ""
    return f"{header}\n{sep}\n{body}{note}"


def _section_items(
    section: str,
    adapter: Any,
    source_id: str,
    source_ids: list[str],
    schema_prefix: str,
    cache: dict,
) -> list[Any]:
    """Items for one section. Encounters/admissions are cached so ``visits``
    reuses them rather than re-querying."""

    def enc() -> list[Any]:
        if "encounters" not in cache:
            cache["encounters"] = _build_encounters_result(adapter, source_ids, schema_prefix, EncountersQuery()).items
        return cache["encounters"]

    def adm() -> list[Any]:
        if "admissions" not in cache:
            # Admissions resolves EMPI inside its own SQL — pass the raw source_id.
            cache["admissions"] = _build_admissions_result(adapter, source_id, schema_prefix, AdmissionsQuery()).items
        return cache["admissions"]

    if section == "demographics":
        return _build_demographics_result(adapter, source_ids, schema_prefix).items
    if section == "encounters":
        return enc()
    if section == "admissions":
        return adm()
    if section == "visits":
        return list(enc()) + list(adm())
    if section == "diagnoses":
        return _build_diagnoses_result(adapter, source_ids, schema_prefix, DiagnosesQuery()).items
    if section == "medications":
        return _build_medications_result(adapter, source_ids, schema_prefix, MedicationsQuery()).items
    if section == "labs":
        return _build_labs_result(adapter, source_ids, schema_prefix, LabsQuery()).items
    if section == "procedures":
        return _build_procedures_result(adapter, source_ids, schema_prefix, ProceduresQuery()).items
    if section == "surgeries":
        return _build_surgeries_result(adapter, source_ids, schema_prefix, SurgeriesQuery()).items
    if section == "imaging":
        return _build_imaging_result(adapter, source_ids, schema_prefix, ImagingQuery()).items
    if section == "immunizations":
        return _build_immunizations_result(adapter, source_ids, schema_prefix, ImmunizationsQuery()).items
    if section == "allergies":
        return _build_allergies_result(adapter, source_ids, schema_prefix, AllergiesQuery()).items
    raise KeyError(f"unknown Patient 360 section: {section}")


def generate_patient360(
    adapter: Any,
    source_id: str,
    *,
    schema_prefix: str = "",
    summarizer: Summarizer,
    sections: Optional[List[str]] = None,
    cache: Optional[SectionCache] = None,
) -> Patient360Result:
    """Run the domain-by-domain pipeline for the selected patient.

    Each section is fetched and (when non-empty) summarized independently; a
    failing section is recorded and skipped rather than aborting the run. A
    final synthesis pass composes the master summary from the section
    narratives. ``summarizer(system_prompt, table_markdown)`` performs the LLM
    call (injected).

    When ``cache`` is provided, a section whose input fingerprint matches the
    stored one reuses the persisted narrative and skips the LLM call
    (fingerprint-based invalidation, Chiron's optimization).
    """
    sections = sections or SECTION_ORDER
    source_ids = federated_source_ids(adapter, source_id, schema_prefix)
    fetch_cache: dict = {}

    out_sections: List[Patient360Section] = []
    narratives: List[tuple[str, str]] = []
    for name in sections:
        try:
            items = _section_items(name, adapter, source_id, source_ids, schema_prefix, fetch_cache)
        except Exception as exc:  # a broken domain must not sink the whole 360
            out_sections.append(Patient360Section(name=name, status="failed", error=f"{type(exc).__name__}: {exc}"))
            continue
        if not items:
            out_sections.append(Patient360Section(name=name, status="empty", row_count=0))
            continue

        markdown = _items_to_markdown(items)
        fingerprint = _fingerprint(markdown)
        narrative: Optional[str] = None
        from_cache = False
        if cache is not None:
            try:
                narrative = cache.get(source_id, name, fingerprint)
                from_cache = narrative is not None
            except Exception:
                narrative = None  # a cache read must never break the run
        if narrative is None:
            try:
                narrative = summarizer(SECTION_PROMPTS[name], markdown)
            except Exception as exc:
                out_sections.append(
                    Patient360Section(
                        name=name, status="failed", row_count=len(items), error=f"{type(exc).__name__}: {exc}"
                    )
                )
                continue
            if cache is not None:
                try:
                    cache.put(source_id, name, fingerprint, narrative)
                except Exception:
                    pass  # persistence failure must not fail the run

        out_sections.append(
            Patient360Section(name=name, status="done", row_count=len(items), narrative=narrative, cached=from_cache)
        )
        narratives.append((name, narrative))

    master_summary = None
    if narratives:
        combined = "\n\n".join(f"### {name}\n{text}" for name, text in narratives)
        try:
            master_summary = summarizer(SYNTHESIS_SYSTEM_PROMPT, combined)
        except Exception:
            master_summary = None

    return Patient360Result(source_id=source_id, sections=out_sections, master_summary=master_summary)
