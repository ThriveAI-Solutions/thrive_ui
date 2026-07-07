"""Code / term normalization shared by the vocab service, ingest loader, and tools."""

from __future__ import annotations


def norm_code(code: str) -> str:
    return code.strip().upper().replace(".", "")


def norm_term(term: str) -> str:
    return " ".join(term.lower().split())
