"""Tests for the Admin Feedback Dashboard."""

from unittest.mock import MagicMock, patch

import pytest

from orm.models import RoleTypeEnum


class TestSetFeedbackTrainingStatus:
    """Tests for the training_status behavior in set_feedback."""

    @pytest.fixture
    def mock_session_state(self):
        """Create a mock session state with messages."""
        with patch("utils.chat_bot_helper.st") as mock_st:
            mock_message = MagicMock()
            mock_message.question = "Test question"
            mock_message.query = "SELECT * FROM test"
            mock_message.feedback = None
            mock_message.training_status = None
            mock_message.feedback_comment = None

            mock_st.session_state.messages = [mock_message]
            mock_st.session_state.cookies = MagicMock()
            yield mock_st, mock_message

    def test_admin_thumbs_up_auto_approves(self, mock_session_state):
        """Admin thumbs up should auto-approve without pending status."""
        from utils.chat_bot_helper import set_feedback

        mock_st, mock_message = mock_session_state
        mock_st.session_state.cookies.get.return_value = "Admin"

        with patch("utils.chat_bot_helper.write_to_file_and_training") as mock_train:
            set_feedback(0, "up")

            # Check training was called
            mock_train.assert_called_once()
            # Check training_status is None (auto-approved)
            assert mock_message.training_status is None
            mock_message.save.assert_called_once()

    def test_non_admin_thumbs_up_sets_pending(self, mock_session_state):
        """Non-admin thumbs up should set training_status to pending."""
        from utils.chat_bot_helper import set_feedback

        mock_st, mock_message = mock_session_state
        mock_st.session_state.cookies.get.return_value = "Doctor"

        with patch("utils.chat_bot_helper.write_to_file_and_training") as mock_train:
            set_feedback(0, "up")

            # Training should NOT be called for non-admin
            mock_train.assert_not_called()
            # Check training_status is pending
            assert mock_message.training_status == "pending"
            mock_message.save.assert_called_once()

    def test_admin_thumbs_down_removes_training(self, mock_session_state):
        """Admin thumbs down should remove from training."""
        from utils.chat_bot_helper import set_feedback

        mock_st, mock_message = mock_session_state
        mock_st.session_state.cookies.get.return_value = "Admin"

        with patch("utils.chat_bot_helper.remove_from_file_training") as mock_remove:
            set_feedback(0, "down", "Test feedback comment")

            mock_remove.assert_called_once()
            assert mock_message.feedback_comment == "Test feedback comment"
            mock_message.save.assert_called_once()

    def test_non_admin_thumbs_down_just_saves(self, mock_session_state):
        """Non-admin thumbs down should just save feedback without training action."""
        from utils.chat_bot_helper import set_feedback

        mock_st, mock_message = mock_session_state
        mock_st.session_state.cookies.get.return_value = "Doctor"

        with patch("utils.chat_bot_helper.write_to_file_and_training") as mock_train:
            with patch("utils.chat_bot_helper.remove_from_file_training") as mock_remove:
                set_feedback(0, "down", "Feedback from doctor")

                mock_train.assert_not_called()
                mock_remove.assert_not_called()
                assert mock_message.feedback == "down"
                assert mock_message.feedback_comment == "Feedback from doctor"
                mock_message.save.assert_called_once()


class TestAdminFeedbackPage:
    """Tests for the admin feedback dashboard page functions."""

    @pytest.fixture
    def mock_admin_session(self):
        """Create a mock session with admin role."""
        with patch("views.admin_feedback.st") as mock_st:
            mock_st.session_state.get.return_value = RoleTypeEnum.ADMIN.value
            mock_st.session_state.cookies = MagicMock()
            mock_st.session_state.cookies.get.return_value = "1"
            yield mock_st

    def test_guard_admin_allows_admin(self, mock_admin_session):
        """Admin users should be allowed through the guard."""
        from views.admin_feedback import _guard_admin

        # Should not raise or call st.stop()
        _guard_admin()
        mock_admin_session.stop.assert_not_called()

    def test_guard_admin_blocks_non_admin(self):
        """Non-admin users should be blocked."""
        with patch("views.admin_feedback.st") as mock_st:
            mock_st.session_state.get.return_value = RoleTypeEnum.DOCTOR.value

            from views.admin_feedback import _guard_admin

            _guard_admin()

            mock_st.error.assert_called_once()
            mock_st.stop.assert_called_once()

    def test_get_status_badge_pending(self):
        """Pending status should return orange badge."""
        from views.admin_feedback import _get_status_badge

        result = _get_status_badge("pending", "up")
        assert ":orange[" in result
        assert "Pending" in result

    def test_get_status_badge_approved(self):
        """Approved status should return green badge."""
        from views.admin_feedback import _get_status_badge

        result = _get_status_badge("approved", "up")
        assert ":green[" in result
        assert "Approved" in result

    def test_get_status_badge_rejected(self):
        """Rejected status should return red badge."""
        from views.admin_feedback import _get_status_badge

        result = _get_status_badge("rejected", "up")
        assert ":red[" in result
        assert "Rejected" in result

    def test_get_status_badge_auto_approved(self):
        """Auto-approved (admin thumbs up with no status) should return blue badge."""
        from views.admin_feedback import _get_status_badge

        result = _get_status_badge(None, "up")
        assert ":blue[" in result
        assert "Auto-Approved" in result

    def test_get_status_badge_thumbs_down(self):
        """Thumbs down with no status should return empty string."""
        from views.admin_feedback import _get_status_badge

        result = _get_status_badge(None, "down")
        assert result == ""


class TestApprovalWorkflow:
    """Tests for the approval and rejection workflow."""

    def test_approve_for_training_success(self):
        """Test that approval updates status and triggers training."""
        mock_message = MagicMock()
        mock_message.question = "Test question"
        mock_message.query = "SELECT * FROM test"

        with patch("views.admin_feedback.SessionLocal") as mock_session_local:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=None)
            mock_session_local.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = mock_message

            with patch("views.admin_feedback.write_to_file_and_training") as mock_train:
                from views.admin_feedback import _approve_for_training

                result = _approve_for_training(1, 2)

                assert result is True
                mock_train.assert_called_once()
                assert mock_message.training_status == "approved"
                assert mock_message.reviewed_by == 2
                mock_session.commit.assert_called_once()

    def test_reject_feedback_success(self):
        """Test that rejection updates status without training."""
        mock_message = MagicMock()

        with patch("views.admin_feedback.SessionLocal") as mock_session_local:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=None)
            mock_session_local.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = mock_message

            from views.admin_feedback import _reject_feedback

            result = _reject_feedback(1, 2)

            assert result is True
            assert mock_message.training_status == "rejected"
            assert mock_message.reviewed_by == 2
            mock_session.commit.assert_called_once()

    def test_approve_nonexistent_message(self):
        """Test that approving a nonexistent message returns False."""
        with patch("views.admin_feedback.SessionLocal") as mock_session_local:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=None)
            mock_session_local.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = None

            with patch("views.admin_feedback.write_to_file_and_training"):
                from views.admin_feedback import _approve_for_training

                result = _approve_for_training(999999, 1)
                assert result is False

    def test_reject_nonexistent_message(self):
        """Test that rejecting a nonexistent message returns False."""
        with patch("views.admin_feedback.SessionLocal") as mock_session_local:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=None)
            mock_session_local.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = None

            from views.admin_feedback import _reject_feedback

            result = _reject_feedback(999999, 1)
            assert result is False


class TestBulkOperations:
    """Tests for bulk approve/reject operations."""

    def test_bulk_approve_calls_individual_approve(self):
        """Bulk approve should call individual approve for each message."""
        from views.admin_feedback import _bulk_approve

        with patch("views.admin_feedback._approve_for_training") as mock_approve:
            mock_approve.return_value = True

            result = _bulk_approve([1, 2, 3], 1)

            assert result == 3
            assert mock_approve.call_count == 3

    def test_bulk_reject_calls_individual_reject(self):
        """Bulk reject should call individual reject for each message."""
        from views.admin_feedback import _bulk_reject

        with patch("views.admin_feedback._reject_feedback") as mock_reject:
            mock_reject.return_value = True

            result = _bulk_reject([1, 2, 3], 1)

            assert result == 3
            assert mock_reject.call_count == 3

    def test_bulk_operations_count_only_successes(self):
        """Bulk operations should count only successful operations."""
        from views.admin_feedback import _bulk_approve

        with patch("views.admin_feedback._approve_for_training") as mock_approve:
            # First two succeed, third fails
            mock_approve.side_effect = [True, True, False]

            result = _bulk_approve([1, 2, 3], 1)

            assert result == 2


class TestKPICard:
    """Tests for the KPI card component."""

    def test_kpi_card_renders_label_and_value(self):
        """KPI card should render with label and value."""
        with patch("views.admin_feedback.st") as mock_st:
            mock_container = MagicMock()
            mock_st.container.return_value.__enter__ = MagicMock(return_value=mock_container)
            mock_st.container.return_value.__exit__ = MagicMock(return_value=None)

            from views.admin_feedback import _kpi_card

            _kpi_card("Test Label", 42)

            mock_st.container.assert_called_once_with(border=True)

    def test_kpi_card_with_help_text(self):
        """KPI card should render help text when provided."""
        with patch("views.admin_feedback.st") as mock_st:
            mock_container = MagicMock()
            mock_st.container.return_value.__enter__ = MagicMock(return_value=mock_container)
            mock_st.container.return_value.__exit__ = MagicMock(return_value=None)

            from views.admin_feedback import _kpi_card

            _kpi_card("Test Label", 42, "Help text here")

            # Just verify it runs without error
            mock_st.container.assert_called_once()
