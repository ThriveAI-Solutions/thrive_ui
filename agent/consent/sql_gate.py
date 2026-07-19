"""Consent gate for freeform run_sql (#244 CON-4).

The curated tools (find_patient, get_patient_clinical_data,
search_patients_by_criteria) are consent-gated. Freeform ``run_sql`` is the
back door: a model-authored query straight against the warehouse would bypass
that gating entirely. Under consent enforcement this gate closes the door.

First-cut posture is deliberately FAIL-CLOSED and simple: any ``run_sql`` that
references a patient-bearing warehouse view is refused when enforcement is on,
steering the model to the curated (gated) tools. It intentionally does NOT try
to distinguish "safe aggregate" from "row-level leak" by inspecting the query
text — that classification is unreliable without a real SQL AST (e.g. a
``WHERE id = 'ok' OR TRUE`` passes a naive check yet returns every row), and a
wrong guess on a disclosure control is worse than a conservative refusal.

``classify_sql`` implements the fuller sqlglot-parsed 3-mode gate:

  - ``non_patient`` — references no patient-bearing view -> run untouched.
  - ``aggregate``   — only COUNT / SUM(CASE .. 0/1) measures over
                      non-identifying GROUP BY keys, applied to EVERY select in
                      the statement (so a clean first UNION arm can't smuggle a
                      row-returning second arm) -> run, then small-cell suppress
                      the measure columns.
  - ``patient_row`` — anything else touching patient data -> refuse.
  - ``unparseable`` — sqlglot could not parse it -> refuse (fail-closed).

This is NOT the ``patient_row`` *rewrite* mode (bind a server-resolved
authorized-id scope into the query). That rewrite needs the federation
id-mapping the curated tools own (the consented roster is internal EMPI
patient_ids; the patient views are keyed by source_id), which cannot be
reconstructed safely in a generic SQL rewrite — so row-level freeform SQL is
REFUSED, not rewritten. Refusal is the correct fail-closed behavior; the model
is steered to the consent-gated curated tools. The classifier's bias is
conservative: any doubt resolves away from ``aggregate``.
"""

from __future__ import annotations

import re
from typing import Optional

import sqlglot
from sqlglot import exp

# Any warehouse view that exposes patient-level rows. Matched case-insensitively
# on word boundaries. The families cover current and future views without an
# exhaustive per-name list (fail-closed toward matching).
_PATIENT_VIEW_RE = re.compile(
    r"\b(?:federated_\w+_v|internal_\w+_v|metric_\w+_v)\b",
    re.IGNORECASE,
)

# Column names that directly or near-directly identify a person. A projection
# or GROUP BY key naming one of these is never "aggregate". Matched on the
# column's bare name, case-insensitively.
_IDENTIFIER_COLUMNS = frozenset(
    {
        "id",
        "patient_id",
        "source_id",
        "cid",
        "empi",
        "empi_id",
        "mrn",
        "medical_record_number",
        "ssn",
        "name",
        "first_name",
        "last_name",
        "middle_name",
        "full_name",
        "display_name",
        "patient_name",
        "dob",
        "date_of_birth",
        "birth_date",
        "address",
        "street",
        "address_line_1",
        "phone",
        "phone_number",
        "email",
        "zip",
        "zip_code",
        "postal_code",
    }
)

# sqlglot dialect names keyed by thrive's adapter dialect taxonomy.
_DIALECT_MAP = {"sqlite": "sqlite", "postgres": "postgres", "redshift": "redshift"}


def references_patient_view(sql: str) -> bool:
    """True if the SQL names any patient-bearing warehouse view."""
    return bool(_PATIENT_VIEW_RE.search(sql or ""))


def _is_identifier_column(name: Optional[str]) -> bool:
    return bool(name) and name.lower() in _IDENTIFIER_COLUMNS


def _projects_only_nonidentifier_columns(node: exp.Expression) -> bool:
    """True if no Column referenced anywhere under ``node`` is an identifier."""
    return not any(_is_identifier_column(col.name) for col in node.find_all(exp.Column))


def _is_approved_measure(node: exp.Expression) -> bool:
    """A count-like measure that leaks no per-person value.

    Approved: ``COUNT(...)`` (any arg — a count is just a number) and
    ``SUM(CASE .. THEN <0/1> .. ELSE <0/1>)``. Everything else (MIN/MAX return
    a real value; AVG/SUM of a raw column, or SUM(CASE) with a non-0/1 literal,
    are single-scope oracles) is rejected.
    """
    if isinstance(node, exp.Count):
        return True
    if isinstance(node, exp.Sum):
        inner = node.this
        if not isinstance(inner, exp.Case):
            return False
        branches = list(inner.args.get("ifs") or [])
        results = [b.args.get("true") for b in branches]
        if inner.args.get("default") is not None:
            results.append(inner.args.get("default"))
        for r in results:
            if not (isinstance(r, exp.Literal) and not r.is_string and r.this in ("0", "1")):
                return False
        return bool(results)
    return False


def _select_is_aggregate_safe(select: exp.Select) -> bool:
    """A single SELECT is aggregate-safe iff every projection is either an
    approved measure or a non-identifier GROUP BY key, there is at least one
    measure, and there is no ``SELECT *``."""
    projections = list(select.expressions)
    if not projections:
        return False

    group = select.args.get("group")
    group_keys = {g.sql() for g in group.expressions} if group else set()

    has_measure = False
    for proj in projections:
        e = proj.unalias() if hasattr(proj, "unalias") else proj
        if isinstance(e, exp.Star) or isinstance(e, exp.Column) and isinstance(e.this, exp.Star):
            return False
        if _is_approved_measure(e):
            has_measure = True
            continue
        # Otherwise it must be a GROUP BY key expression naming no identifier.
        if e.sql() in group_keys and _projects_only_nonidentifier_columns(e):
            continue
        return False
    return has_measure


def _parse(sql: str, dialect: str):
    read = _DIALECT_MAP.get(dialect)
    return sqlglot.parse_one(sql, read=read)


def classify_sql(sql: str, *, dialect: str = "postgres") -> str:
    """Classify freeform SQL into non_patient / aggregate / patient_row / unparseable.

    Conservative: a patient-bearing statement is ``aggregate`` only if EVERY
    SELECT in it (all UNION arms and subqueries included) is aggregate-safe.
    Any parse failure or doubt fails closed toward refusal.
    """
    try:
        tree = _parse(sql or "", dialect)
    except Exception:
        return "unparseable"
    if tree is None:
        return "unparseable"

    tables = [t.name for t in tree.find_all(exp.Table)]
    if not any(references_patient_view(name) for name in tables):
        return "non_patient"

    selects = list(tree.find_all(exp.Select))
    if not selects:
        return "patient_row"
    for select in selects:
        if not _select_is_aggregate_safe(select):
            return "patient_row"
    return "aggregate"


def aggregate_measure_labels(sql: str, *, dialect: str = "postgres") -> frozenset[str]:
    """Output column names that are approved aggregate measures (for suppression).

    Returns the set of result-column names (alias if present, else the measure's
    SQL) that ``classify_sql`` counts as measures, so the caller can small-cell
    suppress exactly those cells and leave GROUP BY label columns intact.
    """
    try:
        tree = _parse(sql or "", dialect)
    except Exception:
        return frozenset()
    if tree is None:
        return frozenset()
    labels: set[str] = set()
    # Only the top-level select's projections become output columns.
    top = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if top is None:
        return frozenset()
    for proj in top.expressions:
        e = proj.unalias() if hasattr(proj, "unalias") else proj
        if _is_approved_measure(e):
            labels.add(proj.alias_or_name)
    return frozenset(labels)
