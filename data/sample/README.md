# Sample Database

Synthetic 1,000-patient EHR/claims dataset that mirrors the production Redshift
`dw` schema (the 17 V1-whitelist views). Used for local agent development and
model comparison. No PHI.

## What's here

- `thrive_sample.sql.zst` — pg_dump-compatible compressed SQL dump. Single
  source of truth, ~5–15 MB. Load via `scripts/load_sample_db.py`.
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
./scripts/synthea/generate.sh                       # ~3–5 min
uv run python -m scripts.sample_db.etl              # ~30 s
git add data/sample/thrive_sample.sql.zst
git commit -m "chore(sample-db): regenerate"
```

See `docs/superpowers/specs/2026-05-13-sample-database-design.md` for the
full design.
