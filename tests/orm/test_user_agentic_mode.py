import pytest
from orm.models import User


def test_user_has_agentic_mode_default_false():
    user = User(username="test@example.com", first_name="Test", last_name="User", password="x")
    assert user.agentic_mode is None or user.agentic_mode is False


def test_user_agentic_mode_in_to_dict():
    user = User(username="test@example.com", first_name="Test", last_name="User", password="x", agentic_mode=True)
    assert user.to_dict()["agentic_mode"] is True
