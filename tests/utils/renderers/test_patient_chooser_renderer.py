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
