import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agent.codes.service import (
    UnknownCodeSetError,
    VocabNotLoadedError,
    expand_sets,
    search_vocab,
)
from orm.models import Base


@pytest.fixture()
def empty_session():
    """A DB with all vocab_* tables created but zero rows -- simulates a fresh
    deploy where the migration ran but the vocab ingest script was never run."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
    engine.dispose()


@pytest.fixture()
def session(vocab_session):
    return vocab_session


def test_tier1_exact_code_match_dot_insensitive(session):
    result = search_vocab(session, vocabulary="icd10", query="e119")
    assert result.codes[0].code == "E11.9"


def test_tier2_set_synonym_returns_set_first(session):
    result = search_vocab(session, vocabulary="icd10", query="DM")
    assert result.sets and result.sets[0].set_id == "dx:diabetes-mellitus"
    assert result.sets[0].member_count == 2
    assert result.sets[0].sample_codes  # non-empty, ≤5


def test_tier3_code_synonym(session):
    result = search_vocab(session, vocabulary="icd10", query="high blood pressure")
    assert any(c.code == "I10" for c in result.codes)


def test_tier3_display_substring(session):
    result = search_vocab(session, vocabulary="icd10", query="hyperglycemia")
    assert any(c.code == "E11.65" for c in result.codes)


def test_limit_respected(session):
    result = search_vocab(session, vocabulary="icd10", query="diabetes", limit=1)
    assert len(result.codes) <= 1


def test_empty_vocabulary_raises_import_script_remedy(session):
    with pytest.raises(VocabNotLoadedError, match="scripts/import_vocab_dump.py") as exc:
        search_vocab(session, vocabulary="snomed", query="diabetes")
    assert "no 'snomed' rows in vocab_codes" in str(exc.value)


def test_expand_sets_returns_member_codes(session):
    codes = expand_sets(session, ["dx:diabetes-mellitus"])
    assert sorted(codes) == ["E11.65", "E11.9"]


def test_expand_sets_unknown_id_raises_with_suggestions(session):
    with pytest.raises(UnknownCodeSetError) as exc:
        expand_sets(session, ["dx:diabetus"])
    assert "dx:diabetes-mellitus" in str(exc.value)


def test_expand_sets_empty_vocab_code_sets_raises_not_loaded(empty_session):
    # Fresh deploy: migration ran, ingest didn't. A set id like this appears
    # verbatim in the tool schema description, so the model can easily pass
    # it -- the error must say to run the import script, not "unknown set".
    with pytest.raises(VocabNotLoadedError, match="scripts/import_vocab_dump.py") as exc:
        expand_sets(empty_session, ["dx:diabetes-mellitus"])
    assert "vocab_code_sets" in str(exc.value)


def test_tier2_exact_set_match_never_displaced_by_prefix_matches(session):
    # 4 candidate sets, cap _MAX_SETS=3: 3 match "flu" only by name prefix
    # (inserted first), 1 matches by exact synonym (inserted last). Exact
    # must win rank 0 and must not be pushed out of the top-3 by the cap.
    result = search_vocab(session, vocabulary="icd10", query="flu")
    assert result.sets, "expected at least one set hit for 'flu'"
    assert result.sets[0].set_id == "vg:INFLUENZA"
    set_ids = {s.set_id for s in result.sets}
    assert "vg:INFLUENZA" in set_ids


def test_like_wildcard_percent_in_query_matches_literally(session):
    result = search_vocab(session, vocabulary="rxnorm", query="5% injectable")
    assert any(c.code == "D5W" for c in result.codes)


def test_like_wildcard_underscore_in_query_does_not_act_as_wildcard(session):
    # "5_" must not match "50mg Tablet" via the underscore-as-any-char LIKE
    # wildcard; the underscore in the user's query is literal.
    result = search_vocab(session, vocabulary="rxnorm", query="5_")
    assert not any(c.code == "RX50" for c in result.codes)
