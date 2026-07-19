"""Small-cell suppression (#244)."""

import pytest

from agent.consent.suppression import (
    SMALL_CELL_THRESHOLD,
    SUPPRESSED_LABEL,
    suppress_cohort,
    suppress_count,
    suppressed_label,
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


def test_default_label_matches_default_threshold():
    # The module default label must describe the module default threshold —
    # they cannot drift (an "11" label under a threshold of 25 misstates the
    # disclosure control).
    assert SUPPRESSED_LABEL == suppressed_label(SMALL_CELL_THRESHOLD)


@pytest.mark.parametrize("threshold", [11, 25, 5])
def test_suppressed_label_describes_threshold(threshold):
    assert suppressed_label(threshold) == f"fewer than {threshold}"


def test_suppress_count_honors_custom_threshold():
    # With a floor of 25, a cell of 11 (safe under the default) is now small.
    assert suppress_count(11, threshold=25) == "fewer than 25"
    assert suppress_count(25, threshold=25) == 25
    assert suppress_count(0, threshold=25) == 0  # empty cell still safe


def test_suppress_cohort_honors_custom_threshold():
    out = suppress_cohort({"a": 11, "b": 300}, additive=False, threshold=25)
    assert out == {"a": "fewer than 25", "b": 300}


def test_suppress_cohort_complementary_uses_custom_threshold_label():
    # Complementary victim must carry the custom-threshold label too.
    out = suppress_cohort({"a": 40, "b": 11, "c": 500}, additive=True, threshold=25)
    assert out["b"] == "fewer than 25"
    assert out["a"] == "fewer than 25"
    assert out["c"] == 500
