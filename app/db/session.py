"""SQLAlchemy engine and session factory."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

_settings = get_settings()

engine = create_engine(
    _settings.sqlalchemy_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=_settings.debug,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a transactional DB session per request.

    Commits on success, rolls back on any exception. Routes that call
    ``db.commit()`` explicitly (uploads, ingestion, feedback) are
    unaffected — the second commit here is a no-op once the transaction
    is already closed. This boundary catches writes that flush without
    an explicit commit — e.g. ``/ask`` persists ``chat_messages`` via
    ``ChatRepository.add_message`` which only flushes; without the
    commit here the rows would roll back at session close and the
    follow-up ``POST /ask/{message_id}/feedback`` would 404.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
