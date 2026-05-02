"""Unit tests for utils.okta_auth and the User-model OIDC columns.

Tests use an in-memory SQLite engine and only exercise pure helpers and the
sync_okta_user_to_db function against a scratch DB. No real OIDC traffic
flows in these tests; the full flow is validated manually against an Okta
Developer org per docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md §11.
"""

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker


def test_user_model_has_okta_sub_and_email_columns():
    """Schema check: User table must expose okta_sub and email columns."""
    from orm.models import Base, User

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    columns = {col["name"]: col for col in inspector.get_columns(User.__tablename__)}

    assert "okta_sub" in columns, "User.okta_sub column missing"
    assert "email" in columns, "User.email column missing"
    # Both must be nullable so existing seeded users keep working.
    assert columns["okta_sub"]["nullable"] is True
    assert columns["email"]["nullable"] is True

    # Both must be unique.
    unique_indexes = inspector.get_unique_constraints(User.__tablename__)
    unique_columns = {col for c in unique_indexes for col in c["column_names"]}
    # SQLAlchemy may register unique=True as either a unique constraint or a
    # unique index depending on backend; check both.
    indexes = inspector.get_indexes(User.__tablename__)
    for idx in indexes:
        if idx.get("unique") and len(idx["column_names"]) == 1:
            unique_columns.add(idx["column_names"][0])

    assert "okta_sub" in unique_columns
    assert "email" in unique_columns
