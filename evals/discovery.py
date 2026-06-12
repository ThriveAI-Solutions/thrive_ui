"""--suggest-patients: read-only probes that find patients with data per
clinical domain, so the roster can be seeded without guessing source_ids.

Approximate by design: a count of rows in the date window, not a clinical
assessment. The human pastes candidates into roster.yaml deliberately.
View/date-column pairs mirror the §7.12 whitelist (tests/agent/redshift_synthetic.sql).
"""

from __future__ import annotations

DOMAIN_PROBES = [
    {"domain": "procedures", "question_ids": ["Q1"], "view": "federated_orders_v", "date_column": "date_of_procedure"},
    {"domain": "labs", "question_ids": ["Q2", "Q3", "Q4"], "view": "federated_results_v", "date_column": "datetime"},
    {"domain": "encounters", "question_ids": ["Q5"], "view": "federated_encounters_v", "date_column": "datetime"},
    {"domain": "immunizations", "question_ids": ["Q6"], "view": "federated_vaccination_v", "date_column": "datetime"},
    {"domain": "documents", "question_ids": ["Q7"], "view": "federated_documents_v", "date_column": "datetime"},
    {"domain": "imaging", "question_ids": ["Q8"], "view": "federated_orders_v", "date_column": "date_of_procedure"},
    {"domain": "medications", "question_ids": ["Q9"], "view": "federated_meds_v", "date_column": "date_prescribed"},
]


def suggest_patients(adapter, date_start: str, date_end: str, per_domain: int = 5) -> dict:
    out: dict[str, list[dict]] = {}
    for probe in DOMAIN_PROBES:
        sql = (
            f"SELECT source_id, COUNT(*) AS record_count "
            f"FROM {adapter.schema_prefix}{probe['view']} "
            f"WHERE {probe['date_column']} >= :date_start AND {probe['date_column']} <= :date_end "
            f"AND source_id IS NOT NULL "
            f"GROUP BY source_id ORDER BY record_count DESC LIMIT {int(per_domain)}"
        )
        try:
            rows = adapter.fetch_all(sql, {"date_start": date_start, "date_end": date_end})
        except Exception as exc:  # a missing view shouldn't kill discovery
            out[probe["domain"]] = [{"error": f"{type(exc).__name__}: {exc}"}]
            continue
        out[probe["domain"]] = [
            {"source_id": str(r["source_id"]), "record_count": int(r["record_count"])} for r in rows
        ]
    return out


def format_roster_snippet(suggestions: dict) -> str:
    domain_questions = {p["domain"]: p["question_ids"] for p in DOMAIN_PROBES}
    per_patient: dict[str, dict] = {}
    for domain, rows in suggestions.items():
        for row in rows:
            if "source_id" not in row:
                continue
            entry = per_patient.setdefault(row["source_id"], {"domains": [], "questions": set()})
            entry["domains"].append(f"{domain}({row['record_count']})")
            entry["questions"].update(domain_questions[domain])
    lines = ["# Suggested candidates — review, then paste under `patients:` in evals/roster.yaml"]
    for source_id, entry in per_patient.items():
        qs = ", ".join(sorted(entry["questions"]))
        domains_str = ", ".join(entry["domains"])
        lines.append(f'  - source_id: "{source_id}"')
        lines.append(f'    label: "auto: {domains_str}"')
        lines.append(f"    # candidate for: {qs}")
    return "\n".join(lines)
