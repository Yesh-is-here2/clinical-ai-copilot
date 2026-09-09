"""Alembic environment for the Clinical AI Copilot backend.

Uses the async recipe (SQLAlchemy's own recommended pattern) so
migrations run over the same ``asyncpg`` driver the application uses,
rather than pulling in a second, sync-only Postgres driver just for
migrations.

Database URL comes from ``app.core.config.settings`` (i.e. the same
``DATABASE_URL`` env var / ``.env`` the FastAPI app reads) rather than a
value hardcoded in ``alembic.ini``, so migrations always target whatever
database the app is actually configured for.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Importing app.models (not an individual model module) populates
# Base.metadata with every table before Alembic inspects it - see the
# docstring in app/models/__init__.py.
from app.core.config import settings
from app.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Override whatever placeholder is in alembic.ini with the real,
# environment-configured database URL. Never hardcode credentials here.
config.set_main_option("sqlalchemy.url", settings.database_url)


def run_migrations_offline() -> None:
    """Generate SQL without opening a database connection.

    This is what `alembic upgrade head --sql` uses, and it's the mode
    used to structurally validate migrations in environments (like CI or
    a sandbox with no route to the real Postgres instance) that cannot
    open a live connection - see LOCAL_INTEGRATION_PENDING.md.
    """

    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live database using an async engine."""

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
