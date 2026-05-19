# scripts/sample_db/transformers/base.py
"""Shared types and helpers for ETL transformers."""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TransformContext:
    """Carried into every transformer. Per-table RNGs avoid cross-coupling."""

    seed: int = 42
    rngs: dict[str, random.Random] = field(default_factory=dict)
    # Output: {table_name: [row_dict, ...]} — transformers append here.
    output: dict[str, list[dict]] = field(default_factory=dict)

    def rng(self, key: str) -> random.Random:
        """Return a deterministic per-key RNG. Independent of other keys."""
        if key not in self.rngs:
            self.rngs[key] = random.Random(self.seed + hash(key) % 10**6)
        return self.rngs[key]

    def add_rows(self, table: str, rows: list[dict]) -> None:
        self.output.setdefault(table, []).extend(rows)


def naive_dt(value) -> dt.datetime:
    """Parse a Synthea timestamp into a UTC-stripped naive datetime.

    Warehouse columns are TIMESTAMP (no zone); strip tzinfo so psycopg2/sqlite
    don't reject a tz-aware value.
    """
    return pd.to_datetime(value, utc=True).to_pydatetime().replace(tzinfo=None)


def str_or_none(value) -> str | None:
    """Return str(value) or None if value is NA. Convenience for optional columns."""
    return str(value) if pd.notna(value) else None
