"""Small-cell suppression (#244)."""

import pytest

from agent.consent.suppression import (
    SUPPRESSED_LABEL,
    suppress_cohort,
    suppress_count,
)


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 0),  # empty cell is safe — not a disclosure
        (1, SUPPRESSED_LABEL),
        (10, SUPPRESSED_LABEL),
        (11, 11),  # threshold is inclusive-safe at 11
        (250, 250),
    ],
)
def test_suppress_count(n, expected):
    assert suppress_count(n) == expected


def test_suppress_count_rejects_bool():
    with pytest.raises(TypeError):
        suppress_count(True)


def test_cohort_suppresses_each_small_cell():
    out = suppress_cohort({"a": 50, "b": 3, "c": 200}, additive=False)
    assert out == {"a": 50, "b": SUPPRESSED_LABEL, "c": 200}


def test_cohort_complementary_suppression_on_single_small_bucket():
    # b is small; a and c are visible. Additive => b recoverable as total-a-c,
    # so the next-smallest visible bucket (a) is also suppressed.
    out = suppress_cohort({"a": 40, "b": 3, "c": 500}, additive=True)
    assert out["b"] == SUPPRESSED_LABEL
    assert out["a"] == SUPPRESSED_LABEL
    assert out["c"] == 500


def test_cohort_no_complementary_when_two_already_suppressed():
    out = suppress_cohort({"a": 2, "b": 4, "c": 500}, additive=True)
    assert out == {"a": SUPPRESSED_LABEL, "b": SUPPRESSED_LABEL, "c": 500}


def test_cohort_non_additive_skips_complementary():
    # Non-additive (e.g. year-over-year) — no subtraction channel to protect.
    out = suppress_cohort({"2024": 3, "2025": 40}, additive=False)
    assert out == {"2024": SUPPRESSED_LABEL, "2025": 40}
