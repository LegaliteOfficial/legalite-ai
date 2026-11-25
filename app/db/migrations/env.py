"""Alembic environment.

Loads target metadata from `app.db.base` so `alembic revision --autogenerate`
sees all ORM models. Uses our app settings for the DB URL so migrations match
runtime configuration.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

import app.db.models  # noqa: F401 — registers all tables on Base.metadata
from app.core.config import get_settings
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject runtime DB URL (overrides empty alembic.ini value).
# configparser uses % for variable interpolation (%(name)s); URL-encoded
# password characters like %40 trigger ValueError("invalid interpolation
# syntax"). Doubling % to %% escapes it so configparser stores the URL
# literally — engine_from_config later reads it back as a single %.
config.set_main_option(
    "sqlalchemy.url",
    get_settings().sqlalchemy_url.replace("%", "%%"),
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
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


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
