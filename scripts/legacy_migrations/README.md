# Legacy SQLite Migration Scripts

These scripts pre-date Alembic. Their schema effects are baked into the
Alembic baseline revision (`alembic/versions/0001_*.py`).

**Do not run these on a new database.** A fresh DB created by the app already
has the final schema; running these would error or no-op.

They are kept here for forensic value:

- `migrate_add_okta_columns.py` is a useful reference for crash-safe SQLite
  table rebuilds (uses SAVEPOINT to widen `username` from VARCHAR(50) to
  VARCHAR(320)). If a future migration needs to do something similar, this is
  a good template.
- The others document the order in which columns/indexes were added during
  development, which can help when reading old commits or DB snapshots.

To make a new schema change, use Alembic:

    uv run alembic revision --autogenerate -m "describe change"
    uv run alembic upgrade head

See the README for the full workflow.
