import datetime as dt

import pandas as pd
import streamlit as st
from sqlalchemy import func, or_

from orm.models import Message, RoleTypeEnum, SessionLocal, User
from utils.enums import MessageType
from utils.vanna_calls import write_to_file_and_training


def _guard_admin():
    """Ensure only admin users can access this page."""
    if st.session_state.get("user_role") != RoleTypeEnum.ADMIN.value:
        st.error("You don't have permission to view this page.")
        st.stop()


def _get_feedback_stats(days: int = 30):
    """Get feedback statistics for the dashboard."""
    since = dt.datetime.now() - dt.timedelta(days=days)

    with SessionLocal() as session:
        # Total feedback counts
        total_thumbs_up = (
            session.query(func.count())
            .select_from(Message)
            .filter(Message.feedback == "up", Message.created_at >= since)
            .scalar()
            or 0
        )
        total_thumbs_down = (
            session.query(func.count())
            .select_from(Message)
            .filter(Message.feedback == "down", Message.created_at >= since)
            .scalar()
            or 0
        )
        pending_approvals = (
            session.query(func.count())
            .select_from(Message)
            .filter(Message.training_status == "pending")
            .scalar()
            or 0
        )
        approved_count = (
            session.query(func.count())
            .select_from(Message)
            .filter(Message.training_status == "approved", Message.created_at >= since)
            .scalar()
            or 0
        )
        rejected_count = (
            session.query(func.count())
            .select_from(Message)
            .filter(Message.training_status == "rejected", Message.created_at >= since)
            .scalar()
            or 0
        )

    return {
        "thumbs_up": total_thumbs_up,
        "thumbs_down": total_thumbs_down,
        "pending": pending_approvals,
        "approved": approved_count,
        "rejected": rejected_count,
    }


def _get_feedback_messages(
    feedback_type: str = "All",
    status_filter: str = "All",
    days: int | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """Get feedback messages with optional filters."""
    with SessionLocal() as session:
        query = (
            session.query(
                Message.id,
                Message.question,
                Message.query.label("sql_query"),
                Message.feedback,
                Message.feedback_comment,
                Message.training_status,
                Message.created_at,
                Message.reviewed_at,
                User.username,
                User.first_name,
                User.last_name,
            )
            .join(User, User.id == Message.user_id)
            .filter(
                Message.type == MessageType.SUMMARY.value,
                Message.feedback.isnot(None),
            )
        )

        # Apply feedback type filter
        if feedback_type == "Thumbs Up":
            query = query.filter(Message.feedback == "up")
        elif feedback_type == "Thumbs Down":
            query = query.filter(Message.feedback == "down")

        # Apply status filter
        if status_filter == "Pending Review":
            query = query.filter(Message.training_status == "pending")
        elif status_filter == "Approved":
            query = query.filter(Message.training_status == "approved")
        elif status_filter == "Rejected":
            query = query.filter(Message.training_status == "rejected")
        elif status_filter == "Auto-Approved (Admin)":
            query = query.filter(
                Message.feedback == "up",
                or_(Message.training_status.is_(None), Message.training_status == ""),
            )

        # Apply date filter
        if days is not None:
            since = dt.datetime.now() - dt.timedelta(days=days)
            query = query.filter(Message.created_at >= since)

        # Get total count for pagination
        total_count = query.count()

        # Apply ordering and pagination
        query = query.order_by(Message.created_at.desc()).offset(offset).limit(limit)

        results = query.all()

    return results, total_count


def _approve_for_training(message_id: int, reviewer_id: int):
    """Approve a message for training."""
    with SessionLocal() as session:
        message = session.query(Message).filter(Message.id == message_id).first()
        if message:
            message.training_status = "approved"
            message.reviewed_by = reviewer_id
            message.reviewed_at = func.now()
            session.commit()

            # Trigger training
            if message.question and message.query:
                new_entry = {"question": message.question, "query": message.query}
                write_to_file_and_training(new_entry)

            return True
    return False


def _reject_feedback(message_id: int, reviewer_id: int):
    """Reject a message (mark as reviewed but not used for training)."""
    with SessionLocal() as session:
        message = session.query(Message).filter(Message.id == message_id).first()
        if message:
            message.training_status = "rejected"
            message.reviewed_by = reviewer_id
            message.reviewed_at = func.now()
            session.commit()
            return True
    return False


def _bulk_approve(message_ids: list, reviewer_id: int):
    """Approve multiple messages for training."""
    success_count = 0
    for msg_id in message_ids:
        if _approve_for_training(msg_id, reviewer_id):
            success_count += 1
    return success_count


def _bulk_reject(message_ids: list, reviewer_id: int):
    """Reject multiple messages."""
    success_count = 0
    for msg_id in message_ids:
        if _reject_feedback(msg_id, reviewer_id):
            success_count += 1
    return success_count


def _kpi_card(label: str, value, help_text: str | None = None):
    """Render a KPI card."""
    c = st.container(border=True)
    with c:
        st.markdown(f"**{label}**")
        st.markdown(f"<h3 style='margin-top:0'>{value}</h3>", unsafe_allow_html=True)
        if help_text:
            st.caption(help_text)


def _get_feedback_icon(feedback: str) -> str:
    """Get icon for feedback type."""
    return "thumbs-up" if feedback == "up" else "thumbs-down"


def _get_status_badge(status: str | None, feedback: str) -> str:
    """Get status badge HTML."""
    if status == "pending":
        return ":orange[Pending Review]"
    elif status == "approved":
        return ":green[Approved]"
    elif status == "rejected":
        return ":red[Rejected]"
    elif feedback == "up":
        return ":blue[Auto-Approved]"
    else:
        return ""


def main():
    _guard_admin()

    st.title("Feedback Dashboard")
    st.caption("Review user feedback and approve non-admin submissions for training")

    # Get current user ID for reviewer tracking
    reviewer_id = None
    try:
        user_id_str = st.session_state.cookies.get("user_id")
        if user_id_str:
            import json

            reviewer_id = json.loads(user_id_str)
    except Exception:
        pass

    # Date range selector
    days_options = {"7 days": 7, "30 days": 30, "90 days": 90, "All time": None}
    days_label = st.segmented_control(
        "Time Range", options=list(days_options.keys()), selection_mode="single", default="30 days"
    )
    days = days_options.get(days_label or "30 days", 30)

    # Stats cards
    stats = _get_feedback_stats(days=days or 365)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _kpi_card("Thumbs Up", stats["thumbs_up"])
    with k2:
        _kpi_card("Thumbs Down", stats["thumbs_down"])
    with k3:
        _kpi_card("Pending", stats["pending"], "Awaiting review")
    with k4:
        _kpi_card("Approved", stats["approved"])
    with k5:
        _kpi_card("Rejected", stats["rejected"])

    st.divider()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        feedback_filter = st.selectbox("Feedback Type", ["All", "Thumbs Up", "Thumbs Down"])
    with col2:
        status_filter = st.selectbox(
            "Status", ["All", "Pending Review", "Approved", "Rejected", "Auto-Approved (Admin)"]
        )
    with col3:
        page_size = st.selectbox("Items per page", [25, 50, 100], index=1)

    # Pagination
    if "feedback_page" not in st.session_state:
        st.session_state.feedback_page = 0

    offset = st.session_state.feedback_page * page_size

    # Get filtered feedback
    feedback_items, total_count = _get_feedback_messages(
        feedback_type=feedback_filter,
        status_filter=status_filter,
        days=days,
        limit=page_size,
        offset=offset,
    )

    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

    # Pagination controls
    if total_count > 0:
        st.caption(f"Showing {offset + 1}-{min(offset + page_size, total_count)} of {total_count} items")

        page_cols = st.columns([1, 1, 3])
        with page_cols[0]:
            if st.button("Previous", disabled=st.session_state.feedback_page == 0):
                st.session_state.feedback_page -= 1
                st.rerun()
        with page_cols[1]:
            if st.button("Next", disabled=st.session_state.feedback_page >= total_pages - 1):
                st.session_state.feedback_page += 1
                st.rerun()

    # Bulk action state
    if "selected_messages" not in st.session_state:
        st.session_state.selected_messages = set()

    # Bulk action buttons (only show if pending items exist in view)
    pending_items = [item for item in feedback_items if item.training_status == "pending"]
    if pending_items:
        bulk_cols = st.columns([1, 1, 3])
        with bulk_cols[0]:
            if st.button("Approve All Pending", type="primary"):
                pending_ids = [item.id for item in pending_items]
                count = _bulk_approve(pending_ids, reviewer_id)
                st.success(f"Approved {count} items for training")
                st.rerun()
        with bulk_cols[1]:
            if st.button("Reject All Pending"):
                pending_ids = [item.id for item in pending_items]
                count = _bulk_reject(pending_ids, reviewer_id)
                st.info(f"Rejected {count} items")
                st.rerun()

    st.divider()

    # Feedback items display
    if not feedback_items:
        st.info("No feedback items found matching your filters.")
    else:
        for item in feedback_items:
            feedback_icon = ":material/thumb_up:" if item.feedback == "up" else ":material/thumb_down:"
            status_badge = _get_status_badge(item.training_status, item.feedback)

            with st.expander(
                f"{feedback_icon} **{item.username}** - {item.created_at.strftime('%Y-%m-%d %H:%M')} {status_badge}",
                expanded=False,
            ):
                # User info
                st.markdown(f"**User:** {item.first_name} {item.last_name} (`{item.username}`)")

                # Question
                st.markdown("**Question:**")
                st.info(item.question or "No question recorded")

                # SQL Query
                if item.sql_query:
                    st.markdown("**SQL Query:**")
                    st.code(item.sql_query, language="sql")

                # Feedback comment (for thumbs down)
                if item.feedback_comment:
                    st.markdown("**Feedback Comment:**")
                    st.warning(item.feedback_comment)

                # Review info
                if item.reviewed_at:
                    st.caption(f"Reviewed at: {item.reviewed_at}")

                # Action buttons (only for pending items)
                if item.training_status == "pending":
                    action_cols = st.columns([1, 1, 3])
                    with action_cols[0]:
                        if st.button("Approve", key=f"approve_{item.id}", type="primary"):
                            if _approve_for_training(item.id, reviewer_id):
                                st.success("Approved and trained!")
                                st.rerun()
                            else:
                                st.error("Failed to approve")
                    with action_cols[1]:
                        if st.button("Reject", key=f"reject_{item.id}"):
                            if _reject_feedback(item.id, reviewer_id):
                                st.info("Rejected")
                                st.rerun()
                            else:
                                st.error("Failed to reject")


if __name__ == "__main__":
    main()
