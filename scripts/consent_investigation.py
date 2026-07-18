"""Count-only consent investigation for #242 (CON-2).

Measures the population consent distribution on ``federated_demographic_v`` —
how many rows carry explicit ``hie_consent = TRUE`` / ``FALSE`` / blank — to
characterise the ~33%-no-consent finding as real signal rather than a mapping
artifact. Emits AGGREGATES ONLY (four integers); it never selects or prints a
patient row, so it is safe to run against prod under the PHI rules.

Usage (from repo root, with .streamlit/secrets.toml pointing [analytics_db] at
the warehouse):

    uv run python -m scripts.consent_investigation

Interpreting the result: if ``consent_false`` is a large, stable share of
non-blank rows, the ~33% is genuine explicit non-consent — not blanks and not a
join dropping TRUE rows. ``consent_blank`` is *unknown*, not a consent state.
Cross-check the person-grain rate by wrapping this against your DISTINCT-patient
resolution; at row grain this is the raw distribution the finding came from.
"""

from __future__ import annotations

import sys

from agent.db.queries.consent import consent_population_counts_sql


def main() -> int:
    from agent.db.analytics_adapter import AnalyticsDbAdapter

    adapter = AnalyticsDbAdapter.from_streamlit_secrets()
    sql, params = consent_population_counts_sql(schema_prefix=adapter.schema_prefix)
    row = adapter.fetch_all(sql, params)[0]

    true_n = int(row["consent_true"] or 0)
    false_n = int(row["consent_false"] or 0)
    blank_n = int(row["consent_blank"] or 0)
    total = int(row["total_rows"] or 0)
    explicit = true_n + false_n

    print("federated_demographic_v consent distribution (row grain, count-only)")
    print(f"  TRUE  : {true_n:>10,}")
    print(f"  FALSE : {false_n:>10,}")
    print(f"  blank : {blank_n:>10,}")
    print(f"  total : {total:>10,}")
    if explicit:
        print(f"  explicit FALSE share: {false_n / explicit:.1%} of non-blank rows")
    if total:
        print(f"  blank share        : {blank_n / total:.1%} of all rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
