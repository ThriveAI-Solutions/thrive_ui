"""DB-backed vocabulary search + code-set expansion (ported from chiron
core/codes/service.py, spec 2026-07-03 §6).

Tiered, first non-empty tier orders the result:
  1. exact code match on code_norm
  2. set name / set synonym (exact, then prefix) → sets block
  3. code synonym / display (exact, prefix, substring)

Chiron's tier 4 (pg_trgm similarity, Postgres-only typo backfill) is
DELETED here: thrive_ui's app SQLite DB is the only place these tables
live (by contract — see CLAUDE.md), and SQLite has no pg_trgm. All
remaining tiers are plain SQLAlchemy.

This module does not create sessions; callers pass a `Session` (chiron's
signature) — thrive_ui's caller will be `ctx.deps.sqlite_session` (later task).
"""

from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agent.codes.normalize import norm_code, norm_term
from orm.models import (
    VocabCode,
    VocabCodeSet,
    VocabCodeSetMember,
    VocabSetSynonym,
    VocabSynonym,
)

_MAX_SETS = 3
_SAMPLE_CODES = 5
_LIKE_ESCAPE = "\\"


class VocabNotLoadedError(RuntimeError):
    pass


class UnknownCodeSetError(ValueError):
    def __init__(self, set_id: str, suggestions: list[str]):
        self.set_id = set_id
        self.suggestions = suggestions
        hint = f"; nearest: {', '.join(suggestions)}" if suggestions else ""
        super().__init__(f"unknown code set {set_id!r}{hint}")


class SetHit(BaseModel):
    set_id: str
    name: str
    member_count: int
    sample_codes: list[str]


class CodeHit(BaseModel):
    code: str
    display_name: str
    vocabulary: str
    is_active: bool


class VocabSearchResult(BaseModel):
    sets: list[SetHit]
    codes: list[CodeHit]


def _escape_like(value: str) -> str:
    """Escape LIKE metacharacters so literal `%`/`_` in user input (e.g. a real
    vocab string like "Dextrose 5% Injectable Solution") can't act as SQL
    wildcards. Pair with `escape=_LIKE_ESCAPE` on every `.like()` call."""
    return (
        value.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2).replace("%", f"{_LIKE_ESCAPE}%").replace("_", f"{_LIKE_ESCAPE}_")
    )


def _hit(row: VocabCode) -> CodeHit:
    return CodeHit(code=row.code, display_name=row.display, vocabulary=row.vocabulary, is_active=row.is_active)


def _set_hits(session: Session, needle: str, vocabulary: str) -> list[SetHit]:
    # exact synonym/name first, then prefix — two passes, deduped by set_id,
    # so a prefix-only match can never displace an exact match once _MAX_SETS
    # candidates have been collected (mirrors the tier-3 cascade below).
    name_l = func.lower(VocabCodeSet.name)
    needle_like = _escape_like(needle)
    exact = (
        select(VocabCodeSet)
        .outerjoin(VocabSetSynonym)
        .where((VocabSetSynonym.term_norm == needle) | (name_l == needle))
        .distinct()
    )
    prefix = (
        select(VocabCodeSet)
        .outerjoin(VocabSetSynonym)
        .where(
            VocabSetSynonym.term_norm.like(f"{needle_like}%", escape=_LIKE_ESCAPE)
            | name_l.like(f"{needle_like}%", escape=_LIKE_ESCAPE)
        )
        .distinct()
    )
    hits: list[SetHit] = []
    seen_set_ids: set[str] = set()
    for stmt in (exact, prefix):
        for cs in session.scalars(stmt):
            if cs.set_id in seen_set_ids:
                continue
            seen_set_ids.add(cs.set_id)
            members = (
                select(VocabCode.code)
                .join(VocabCodeSetMember, VocabCodeSetMember.code_id == VocabCode.id)
                .where(VocabCodeSetMember.set_id == cs.set_id, VocabCode.vocabulary == vocabulary)
            )
            codes = list(session.scalars(members))
            if not codes:
                continue
            hits.append(
                SetHit(set_id=cs.set_id, name=cs.name, member_count=len(codes), sample_codes=codes[:_SAMPLE_CODES])
            )
            if len(hits) >= _MAX_SETS:
                return hits
    return hits


def search_vocab(session: Session, *, vocabulary: str, query: str, limit: int = 15) -> VocabSearchResult:
    total = session.scalar(select(func.count()).select_from(VocabCode).where(VocabCode.vocabulary == vocabulary))
    if not total:
        raise VocabNotLoadedError(
            f"no {vocabulary!r} rows in vocab_codes — run "
            "`uv run python scripts/import_vocab_dump.py data/vocab/` to load the vocabulary tables"
        )

    needle_term = norm_term(query)
    needle_code = norm_code(query)
    base = select(VocabCode).where(VocabCode.vocabulary == vocabulary)

    # Tier 1: exact code
    if needle_code:
        exact = list(session.scalars(base.where(VocabCode.code_norm == needle_code)))
        if exact:
            return VocabSearchResult(sets=[], codes=[_hit(r) for r in exact[:limit]])

    # Tier 2: sets
    sets = _set_hits(session, needle_term, vocabulary) if needle_term else []

    # Tier 3: synonym / display — exact, prefix, substring; dedup preserving rank
    ranked: list[VocabCode] = []
    seen: set[int] = set()
    if needle_term:
        display_l = func.lower(VocabCode.display)
        needle_like = _escape_like(needle_term)
        syn = select(VocabCode).join(VocabSynonym).where(VocabCode.vocabulary == vocabulary)
        tiers = [
            syn.where(VocabSynonym.term_norm == needle_term),
            base.where(display_l == needle_term),
            syn.where(VocabSynonym.term_norm.like(f"{needle_like}%", escape=_LIKE_ESCAPE)),
            base.where(display_l.like(f"{needle_like}%", escape=_LIKE_ESCAPE)),
            syn.where(VocabSynonym.term_norm.like(f"%{needle_like}%", escape=_LIKE_ESCAPE)),
            base.where(display_l.like(f"%{needle_like}%", escape=_LIKE_ESCAPE)),
        ]
        for stmt in tiers:
            for row in session.scalars(stmt.order_by(VocabCode.code).limit(limit)):
                if row.id not in seen:
                    seen.add(row.id)
                    ranked.append(row)
            if len(ranked) >= limit:
                break

    return VocabSearchResult(sets=sets, codes=[_hit(r) for r in ranked[:limit]])


def expand_sets(session: Session, set_ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for set_id in set_ids:
        if session.get(VocabCodeSet, set_id) is None:
            total_sets = session.scalar(select(func.count()).select_from(VocabCodeSet))
            if not total_sets:
                raise VocabNotLoadedError(
                    "vocab_code_sets is empty — run "
                    "`uv run python scripts/import_vocab_dump.py data/vocab/` to load the vocabulary tables"
                )
            prefix = _escape_like(set_id.split(":")[0])
            suggestions = list(
                session.scalars(
                    select(VocabCodeSet.set_id)
                    .where(VocabCodeSet.set_id.like(f"{prefix}:%", escape=_LIKE_ESCAPE))
                    .limit(5)
                )
            )
            raise UnknownCodeSetError(set_id, suggestions)
        members = (
            select(VocabCode.code)
            .join(VocabCodeSetMember, VocabCodeSetMember.code_id == VocabCode.id)
            .where(VocabCodeSetMember.set_id == set_id)
        )
        for code in session.scalars(members):
            if code not in seen:
                seen.add(code)
                out.append(code)
    return out
