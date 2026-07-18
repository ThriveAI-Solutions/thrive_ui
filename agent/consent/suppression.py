"""Small-cell suppression for the de-identified cohort/aggregate exemption.

HealtheLink confirmed aggregate/population queries do NOT require consent
filtering, but a small count is a re-identification channel even when
de-identified. So every count-like cell below a threshold is replaced with a
fixed label *in the data plane* — before the model, any result cache, or the UI
ever sees the raw number. A suppressed cell is a literal string, never a
rounded or fuzzed number (rounding still leaks the band).

Threshold: DEFAULT 11 (a cell of 1..10 is suppressed; 0 and >=11 pass). This is
the value both Chiron and the thrive meeting notes converged on informally, but
it is NOT yet ratified by HeL — flagged for sign-off. Change ``SMALL_CELL_THRESHOLD``
in one place if HeL sets a different floor.

Complementary suppression: in an *additive* breakdown (buckets sum to a visible
total), suppressing exactly one bucket is pointless — it's recoverable as
``total - sum(other visible buckets)``. So if exactly one bucket is suppressed,
the next-smallest visible bucket is suppressed too. Non-additive breakdowns
(buckets not mutually exclusive against the total) have no subtraction channel
and skip this rule.
"""

from __future__ import annotations

from typing import Union

# Flagged for HeL ratification — informal "~11 / single-digit-%" heuristic.
SMALL_CELL_THRESHOLD = 11
SUPPRESSED_LABEL = "fewer than 11"

Cell = Union[int, str]


def suppress_count(n: int) -> Cell:
    """Suppress a single count cell: 1..threshold-1 -> label; 0 and >=threshold pass."""
    if isinstance(n, bool):  # bool is an int subclass; never a count
        raise TypeError("suppress_count expects an integer count, not a bool")
    if 0 < n < SMALL_CELL_THRESHOLD:
        return SUPPRESSED_LABEL
    return n


def suppress_cohort(buckets: dict[str, int], *, additive: bool = True) -> dict[str, Cell]:
    """Suppress every small cell in a breakdown, applying the complementary rule.

    ``buckets`` maps label -> count. With ``additive=True`` (buckets partition a
    visible total), if exactly one bucket falls in the suppression band, the
    next-smallest *un-suppressed* bucket is suppressed too, so the first isn't
    recoverable by subtraction. ``additive=False`` skips the complementary step.
    """
    out: dict[str, Cell] = {k: suppress_count(v) for k, v in buckets.items()}

    if not additive:
        return out

    suppressed_keys = [k for k, v in out.items() if v == SUPPRESSED_LABEL]
    if len(suppressed_keys) == 1:
        # Suppress the next-smallest still-visible bucket (largest count first
        # would over-suppress; smallest visible closes the subtraction channel
        # with the least information loss).
        visible = {k: v for k, v in out.items() if v != SUPPRESSED_LABEL and isinstance(v, int)}
        if visible:
            victim = min(visible, key=lambda k: visible[k])
            out[victim] = SUPPRESSED_LABEL
    return out
