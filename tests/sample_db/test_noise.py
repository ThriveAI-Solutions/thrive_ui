import random

from scripts.sample_db.noise import (
    code_type_variants_for,
    inject_polyglot_code_types,
    pick_code_type,
)


def test_pick_code_type_returns_canonical_string():
    rng = random.Random(42)
    out = pick_code_type("RxNorm", rng, empty_rate=0.0)
    assert out in {
        "RxNorm",
        "RXNORM",
        "RXNorm",
        "NLM RxNorm",
        "2.16.840.1.113883.6.88",
    }


def test_pick_code_type_can_return_empty():
    rng = random.Random(42)
    samples = [pick_code_type("LOINC", rng, empty_rate=0.5) for _ in range(1000)]
    empties = sum(1 for s in samples if s == "")
    # Within tolerance.
    assert 400 <= empties <= 600


def test_variants_for_unknown_returns_singleton():
    assert code_type_variants_for("UNKNOWN_SYSTEM") == ["UNKNOWN_SYSTEM"]


def test_inject_polyglot_overwrites_column():
    rng = random.Random(42)
    rows = [{"code_type": "RxNorm"} for _ in range(100)]
    inject_polyglot_code_types(rows, "code_type", "RxNorm", rng, empty_rate=0.0)
    distinct = {r["code_type"] for r in rows}
    # We should see at least 2 spellings out of 100 rows.
    assert len(distinct) >= 2
    assert all(r["code_type"] != "" for r in rows)


def test_deterministic_given_seed():
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    a = [pick_code_type("RxNorm", rng_a, empty_rate=0.3) for _ in range(50)]
    b = [pick_code_type("RxNorm", rng_b, empty_rate=0.3) for _ in range(50)]
    assert a == b
