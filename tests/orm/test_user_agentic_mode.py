from orm.models import User, UserRole


def test_user_agentic_mode_defaults_on(in_memory_orm_session):
    """A user inserted without specifying agentic_mode gets the column
    default, which is now True (agentic mode on for everybody)."""
    with in_memory_orm_session() as session:
        role_id = session.query(UserRole).filter(UserRole.role_name == "Doctor").one().id
        user = User(
            username="test@example.com",
            first_name="Test",
            last_name="User",
            password="x",
            user_role_id=role_id,
            email="test@example.com",  # required per Epic #179
            organization="TestOrg",
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        assert user.agentic_mode is True


def test_user_agentic_mode_in_to_dict():
    user = User(
        username="test@example.com",
        first_name="Test",
        last_name="User",
        password="x",
        agentic_mode=True,
        email="test@example.com",
        organization="TestOrg",
    )
    assert user.to_dict()["agentic_mode"] is True
