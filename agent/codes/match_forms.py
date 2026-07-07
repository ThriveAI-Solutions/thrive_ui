"""Dotted/undotted ICD code expansion for warehouse queries.

Code forms expansion: ICD codes may appear in the warehouse both dotted
(e.g. E11.9) and undotted (e.g. E119); the code_match_forms function
generates all forms so that exact IN-matching doesn't silently miss rows.
"""

from __future__ import annotations

import re


_ICD_SHAPE = re.compile(r"^[A-Z]\d")  # ICD-10 letter+digit; ICD-9 handled by digit branch


def _dot_forms(undotted: str) -> list[str]:
    """Insert the canonical dot(s) for ICD-shaped codes; [] when no dot form applies.

    Returns a list because one shape is genuinely ambiguous (see below); every
    other shape returns at most one candidate.
    """
    if len(undotted) <= 3:
        return []
    if undotted.isdigit():
        # Pure-numeric: ICD-9 codes are <=5 digits; SNOMED concept ids are >=6
        # digits, so disambiguate on length rather than inventing a phantom dot.
        return [undotted[:3] + "." + undotted[3:]] if len(undotted) <= 5 else []
    if undotted.startswith("E") and undotted[1:].isdigit() and len(undotted) > 4:
        # Ambiguous shape: an undotted E-code of len>4 could be ICD-10
        # (dot-after-3, e.g. E1169 -> E11.69) or ICD-9 (dot-after-4, e.g.
        # E8500 -> E850.0) — the two vocabularies overlap on this letter and
        # neither dotting is inferable from the string alone. Emit BOTH
        # candidates; the wrong one matches nothing (harmless but noisy),
        # same tolerance as the pure-numeric SNOMED case above.
        return [undotted[:3] + "." + undotted[3:], undotted[:4] + "." + undotted[4:]]
    if _ICD_SHAPE.match(undotted) or undotted[0] in "VE":
        return [undotted[:3] + "." + undotted[3:]]
    return []


def code_match_forms(codes: list[str]) -> list[str]:
    """Expand each code into every warehouse spelling: as-given, undotted, dotted.

    Survey note (not a per-code guarantee): the dw was observed to store some
    codes both dotted and undotted (e.g. E11.9 vs E119) — exact IN-matching
    must carry both forms since not every row is known to have both spellings.
    Non-ICD codes (SNOMED numerics) pass through untouched — inventing
    732.11009 would be a phantom (harmless but noisy).
    """
    out: list[str] = []
    for raw in codes:
        c = raw.strip().upper()
        undotted = c.replace(".", "")
        dot_forms = [c] if "." in c else _dot_forms(undotted)
        for form in (c, undotted, *dot_forms):
            if form and form not in out:
                out.append(form)
    return out
