"""Developer integration smoke test for the user-questions queries.

This exercises get_user_recent_questions / get_user_questions_page against the
local dev SQLite DB (./pgDatabase/db.sqlite3) and a known seeded user. It is
inherently non-hermetic, so it skips cleanly when that DB or user isn't present
(e.g. CI or a fresh checkout) rather than failing the suite.
"""

from pathlib import Path

import pytest

from orm.functions import get_user_questions_page, get_user_recent_questions
from orm.models import SessionLocal, User


def test_recent_questions_for_known_user_kr():
    db_file = Path("./pgDatabase/db.sqlite3").resolve()
    if not db_file.exists():
        pytest.skip(f"Local SQLite DB not present at {db_file}; developer-only smoke test")

    # orm.models.SessionLocal is already bound to the configured DB at import,
    # so no secrets mutation is needed (and st.secrets is read-only anyway).
    with SessionLocal() as session:
        user = session.query(User).filter(User.username == "thriveai-kr").first()
        if user is None:
            pytest.skip("User 'thriveai-kr' not present in local DB; developer-only smoke test")
        user_id = user.id

    # Recent questions (deduped)
    recent = get_user_recent_questions(user_id, limit=200)
    assert isinstance(recent, list)

    # Paged questions (with status/elapsed)
    page = get_user_questions_page(user_id, page=1, page_size=50)
    assert isinstance(page, dict)
    assert "items" in page and "total" in page
    items = page["items"]
    assert isinstance(items, list)
    # When there is history for this user, we expect well-formed rows.
    if items:
        row = items[0]
        assert "question" in row and "created_at" in row
        assert "status" in row and "elapsed_seconds" in row
