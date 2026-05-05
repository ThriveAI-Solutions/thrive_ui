from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from orm.models import Base, _get_database_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Resolve URL from app secrets (or THRIVE_SQLITE_PATH env var fallback) so
# that running `alembic` from the CLI hits the same DB the app uses.
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", _get_database_url())

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
