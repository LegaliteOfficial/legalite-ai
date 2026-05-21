"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Provide safe defaults so tests can import settings without a real .env.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("APP_ENV", "development")


@pytest.fixture()
def settings_env() -> Iterator[None]:
    """Marker fixture for tests that depend on env-derived settings."""
    yield
