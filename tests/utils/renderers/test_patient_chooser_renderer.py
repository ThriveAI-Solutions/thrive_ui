import json
from unittest.mock import MagicMock, patch

from utils.renderers.patient_chooser import render_patient_chooser


def test_chooser_renders_button_per_match():
    msg = MagicMock()
    msg.content = json.dumps(
        {
            "matches": [
                {
                    "source_id": "src-1",
                    "display_name": "John Smith",
                    "dob": "1962-05-01",
                    "facilities_seen": ["Buffalo Medical Group"],
                    "most_recent_activity": "2026-04-01",
                },
                {
                    "source_id": "src-2",
                    "display_name": "John Smith",
                    "dob": "1971-08-12",
                    "facilities_seen": ["Kaleida"],
                    "most_recent_activity": "2026-03-15",
                },
            ],
            "total_unique": 2,
            "truncated": False,
        }
    )
    with patch("utils.renderers.patient_chooser.st") as st:
        st.button = MagicMock(return_value=False)
        render_patient_chooser(msg, index=0)
        # One button per match (plus possibly a "see more" if truncated)
        assert st.button.call_count == 2


def test_chooser_button_keys_use_message_id_not_index():
    """Keys must remain stable when older messages are trimmed and
    `index` shifts. message.id is the persisted PK and never moves.
    """
    msg = MagicMock()
    msg.id = 4242
    msg.content = json.dumps(
        {
            "matches": [
                {"source_id": "src-a", "display_name": "A", "dob": None, "facilities_seen": []},
                {"source_id": "src-b", "display_name": "B", "dob": None, "facilities_seen": []},
            ],
            "total_unique": 2,
            "truncated": False,
        }
    )
    with patch("utils.renderers.patient_chooser.st") as st:
        st.button = MagicMock(return_value=False)
        render_patient_chooser(msg, index=7)
        keys_used = [call.kwargs["key"] for call in st.button.call_args_list]
        assert all("4242" in k for k in keys_used), keys_used
        assert all("index-7" not in k for k in keys_used), keys_used


def test_chooser_button_click_writes_session_state():
    msg = MagicMock()
    msg.content = json.dumps(
        {
            "matches": [
                {
                    "source_id": "src-clicked",
                    "display_name": "John Smith",
                    "dob": "1962-05-01",
                    "facilities_seen": [],
                    "most_recent_activity": None,
                }
            ],
            "total_unique": 1,
            "truncated": False,
        }
    )
    fake_session = {}
    with patch("utils.renderers.patient_chooser.st") as st:
        # First button click returns True
        st.button = MagicMock(return_value=True)
        st.session_state = fake_session
        render_patient_chooser(msg, index=0)
        assert fake_session["selected_patient_source_id"] == "src-clicked"


def test_chooser_button_click_retriggers_pending_question():
    """When the user clicks a patient, the chat loop must re-fire the
    original question with the slot now filled. The handler does this
    by copying pending_user_question into my_question — that's the
    same trigger chat_input uses for new questions."""
    msg = MagicMock()
    msg.content = json.dumps(
        {
            "matches": [
                {
                    "source_id": "src-clicked",
                    "display_name": "John Smith",
                    "dob": "1962-05-01",
                    "facilities_seen": [],
                    "most_recent_activity": None,
                }
            ],
            "total_unique": 1,
            "truncated": False,
        }
    )
    fake_session = {
        "pending_user_question": "Has John Smith had any procedures?",
    }
    with patch("utils.renderers.patient_chooser.st") as st:
        st.button = MagicMock(return_value=True)
        st.session_state = fake_session
        render_patient_chooser(msg, index=0)
        assert fake_session["my_question"] == "Has John Smith had any procedures?"


def test_chooser_button_click_with_no_pending_question_is_safe():
    """If somehow there's no stashed question (e.g. session restart with
    an old chooser still in history), clicking shouldn't crash — it just
    sets the slot and lets the user re-ask."""
    msg = MagicMock()
    msg.content = json.dumps(
        {
            "matches": [
                {
                    "source_id": "src-clicked",
                    "display_name": "X",
                    "dob": None,
                    "facilities_seen": [],
                }
            ],
            "total_unique": 1,
            "truncated": False,
        }
    )
    fake_session = {}
    with patch("utils.renderers.patient_chooser.st") as st:
        st.button = MagicMock(return_value=True)
        st.session_state = fake_session
        render_patient_chooser(msg, index=0)
    assert fake_session["selected_patient_source_id"] == "src-clicked"
    # my_question should not be set to a junk value when there's nothing to resume.
    assert fake_session.get("my_question") in (None,)
