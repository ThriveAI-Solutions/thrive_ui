from datetime import date

import pytest

from evals.patients import resolve_patient


def test_resolves_known_source_id(synthetic_adapter):
    sp = resolve_patient(synthetic_adapter, "src-john-1962")
    assert sp.source_id == "src-john-1962"
    assert sp.display_name == "John Smith"
    assert sp.dob == date(1962, 5, 1)
    assert sp.selection_origin == "user_click"


def test_unknown_source_id_raises(synthetic_adapter):
    with pytest.raises(LookupError, match="src-nope"):
        resolve_patient(synthetic_adapter, "src-nope")
