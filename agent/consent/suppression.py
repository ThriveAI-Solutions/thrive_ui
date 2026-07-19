"""Small-cell suppression for the de-identified cohort/aggregate exemption.

HealtheLink confirmed aggregate/population queries do NOT require consent
filtering, but a small count is a re-identification channel even when
de-identified. So every count-like cell below a threshold is replaced with a
fixed label *in the data plane* — before the model, any result cache, or the UI
ever sees the raw number. A suppressed cell is a literal string, never a
rounded or fuzzed number (rounding still leaks the band).

Threshold: default 11 (a cell of 1..10 is suppressed; 0 and >=11 pass). 11 is
the HHS/HIPAA-adjacent "cell size < 11" small-cell convention and matches
Chiron, but for thrive it is NOT yet ratified — the thrive meeting notes only
record the *general principle* that single-digit cells are re-identification
risks, never a signed-off value (Sarah/HeL own the consent-policy sign-off).
So the threshold is a config value: ``[security].small_cell_threshold`` (read
at the tool boundary, default ``SMALL_CELL_THRESHOLD``) and every function
takes it as a parameter. Do NOT re-describe 11 as "decided" until HeL ratifies
it. The suppressed-cell label is derived from the active threshold
(``suppressed_label``) so an "11" label can never appear under a different
floor. See #312 (threshold config) for the ratification thread.

Complementary suppression: in an *additive* breakdown (buckets sum to a visible
total), suppressing exactly one bucket is pointless — it's recoverable as
``total - sum(other visible buckets)``. So if exactly one bucket is suppressed,
the next-smallest visible bucket is suppressed too. Non-additive breakdowns
(buckets not mutually exclusive against the total) have no subtraction channel
and skip this rule.
"""

from __future__ import annotations

from typing import Union

# Default floor: HHS/HIPAA-adjacent "cell size < 11" small-cell convention.
# INFORMAL for thrive — pending Sarah/HeL ratification (#312). Overridable via
# [security].small_cell_threshold; passed explicitly into every function here.
SMALL_CELL_THRESHOLD = 11

Cell = Union[int, str]


def suppressed_label(threshold: int) -> str:
    """The label a suppressed cell renders as, describing the active floor.

    Derived from the threshold so it can never misstate the control (an "11"
    label under a floor of 25 would understate the suppression band).
    """
    return f"fewer than {threshold}"


# Label for the module-default threshold. Kept as a module constant for the
# common default-floor call sites and tests; custom floors use suppressed_label.
SUPPRESSED_LABEL = suppressed_label(SMALL_CELL_THRESHOLD)


def suppress_count(n: int, *, threshold: int = SMALL_CELL_THRESHOLD) -> Cell:
    """Suppress a single count cell: 1..threshold-1 -> label; 0 and >=threshold pass."""
    if isinstance(n, bool):  # bool is an int subclass; never a count
        raise TypeError("suppress_count expects an integer count, not a bool")
    if 0 < n < threshold:
        return suppressed_label(threshold)
    return n


def suppress_cohort(
    buckets: dict[str, int], *, additive: bool = True, threshold: int = SMALL_CELL_THRESHOLD
) -> dict[str, Cell]:
    """Suppress every small cell in a breakdown, applying the complementary rule.

    ``buckets`` maps label -> count. With ``additive=True`` (buckets partition a
    visible total), if exactly one bucket falls in the suppression band, the
    next-smallest *un-suppressed* bucket is suppressed too, so the first isn't
    recoverable by subtraction. ``additive=False`` skips the complementary step.
    """
    label = suppressed_label(threshold)
    out: dict[str, Cell] = {k: suppress_count(v, threshold=threshold) for k, v in buckets.items()}

    if not additive:
        return out

    suppressed_keys = [k for k, v in out.items() if v == label]
    if len(suppressed_keys) == 1:
        # Suppress the next-smallest still-visible bucket (largest count first
        # would over-suppress; smallest visible closes the subtraction channel
        # with the least information loss).
        visible = {k: v for k, v in out.items() if v != label and isinstance(v, int)}
        if visible:
            victim = min(visible, key=lambda k: visible[k])
            out[victim] = label
    return out
