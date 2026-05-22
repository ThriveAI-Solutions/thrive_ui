# Sample Database

Synthetic EHR/claims dataset that mirrors the production Redshift
`dw` schema (the 17 V1-whitelist views). Used for local agent development and
model comparison. No PHI.

Current committed dump: **237 patients** (`SAMPLE_POPULATION=200` seed; Synthea
generates extra family members). Spans ~10 years of history, ~107k lab results,
~17k encounters, ~16k medications, ~10k problems, ~72k rollup events.

## What's here

- `thrive_sample.sql.zst` — pg_dump-compatible compressed SQL dump. Single
  source of truth, ~5 MB. Load via `scripts/load_sample_db.py`.
- `synthea/` — (gitignored) raw Synthea CSV outputs. Regenerable.

## Loading

```bash
docker compose up -d
uv run python scripts/load_sample_db.py            # Postgres (default)
uv run python scripts/load_sample_db.py --target sqlite --path ./sample.db
```

## Regenerating (rare — only when the schema or noise model changes)

Requires Java 11+ installed locally.

```bash
SAMPLE_POPULATION=200 ./scripts/synthea/generate.sh   # ~1–2 min for 200
uv run python -m scripts.sample_db.etl                # ~5–10 min
git add data/sample/thrive_sample.sql.zst
git commit -m "chore(sample-db): regenerate"
```

A larger population (e.g. 1000) is supported but produces a much larger dump
(~50–100 MB compressed) and the ETL takes 30+ minutes — push to LFS if you
go that route.

See `docs/superpowers/specs/2026-05-13-sample-database-design.md` for the
full design.
