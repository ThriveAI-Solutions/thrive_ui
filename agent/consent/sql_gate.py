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

The fuller design (offered by the sister project as an option, not shipped
here): a sqlglot-parsed 3-mode gate — ``non_patient`` untouched, ``aggregate``
run then small-cell-suppressed, ``patient_row`` rewritten to bind a
server-resolved authorized-id scope into the query structure — plus a
consented-patient allow-list snapshot to bind against. That needs both a parser
dependency and the snapshot infra thrive doesn't have yet; tracked as the
remaining #244 work.
"""

from __future__ import annotations

import re

# Any warehouse view that exposes patient-level rows. Matched case-insensitively
# on word boundaries. The families cover current and future views without an
# exhaustive per-name list (fail-closed toward matching).
_PATIENT_VIEW_RE = re.compile(
    r"\b(?:federated_\w+_v|internal_\w+_v|metric_\w+_v)\b",
    re.IGNORECASE,
)


def references_patient_view(sql: str) -> bool:
    """True if the SQL names any patient-bearing warehouse view."""
    return bool(_PATIENT_VIEW_RE.search(sql or ""))
