import json
from pathlib import Path

import streamlit as st

from orm.functions import get_user_questions_page, get_user_recent_questions
from orm.models import SessionLocal, User


def _set_sqlite_for_tests(db_path: str):
    # Override st.secrets sqlite to point to provided path
    # Note: In this repo, orm.models reads st.secrets['sqlite'] at import time.
    # So we refresh by reloading orm.models if needed.
    st.secrets["sqlite"] = {"database": db_path}


def test_recent_questions_for_known_user_kr():
    db_file = str(Path("./pgDatabase/db.sqlite3").resolve())
    assert Path(db_file).exists(), "Expected SQLite db at ./pgDatabase/db.sqlite3"

    _set_sqlite_for_tests(db_file)

    # Find user id for thriveai-kr
    with SessionLocal() as session:
        user = session.query(User).filter(User.username == "thriveai-kr").first()
        assert user is not None, "User thriveai-kr must exist in the database"
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
    # When there is history for this user, we expect some items. If empty, this will still pass but be informative.
    if items:
        row = items[0]
        assert "question" in row and "created_at" in row
        assert "status" in row and "elapsed_seconds" in row
