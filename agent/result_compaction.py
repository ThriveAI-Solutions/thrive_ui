"""Compaction of list-shaped tool results before they're shown to the LLM.

Tool results that carry a list of structured rows (ClinicalResult.items,
CohortResult.sample, etc.) serialize to JSON for the LLM with every nullable
field present as `"x": null` on every row. For weak/local models that is
unnecessary noise — and it costs context tokens that could go toward the
agent's reply.

`compact_rows` removes fields that are None across every row (column-level
prune; row count is preserved). `CompactingListResult` is a Pydantic v2
mixin that applies the prune at `model_dump` time to a single named list
field, so in-process Python access (attribute reads, DataFrame adapters)
sees the un-compacted typed model while the LLM-visible JSON is slim.
"""

from __future__ import annotations
from typing import Any, ClassVar

from pydantic import BaseModel, model_serializer


def compact_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop keys that are None across every row.

    Returns (cleaned_rows, dead_keys). Empty input or no dead keys returns
    the original list unchanged and an empty dead-key list.
    """
    if not rows:
        return rows, []
    all_keys: set[str] = set().union(*(r.keys() for r in rows))
    dead = sorted(k for k in all_keys if all(r.get(k) is None for r in rows))
    if not dead:
        return rows, []
    dead_set = set(dead)
    cleaned = [{k: v for k, v in r.items() if k not in dead_set} for r in rows]
    return cleaned, dead


class CompactingListResult(BaseModel):
    """Pydantic mixin: prune null-only fields from one designated list field
    at JSON-serialization time. Subclasses set ``_list_field``.

    The Python attribute stays intact — only the dict produced by
    ``model_dump`` / ``model_dump_json`` is compacted. Adapters that read
    ``result.<list_field>`` get the typed objects untouched.
    """

    _list_field: ClassVar[str] = ""

    @model_serializer(mode="wrap")
    def _serialize_compacted(self, handler) -> dict[str, Any]:
        data = handler(self)
        field = self.__class__._list_field
        if not field:
            return data
        items = data.get(field)
        if not isinstance(items, list) or not items:
            return data
        if not all(isinstance(row, dict) for row in items):
            return data
        cleaned, dead = compact_rows(items)
        if not dead:
            return data
        data[field] = cleaned
        data["omitted_empty_fields"] = dead
        return data
