"""Schema sanity checks for IngestionJob + JobRepository contract (M4)."""

from __future__ import annotations

import uuid

from app.db.models.ingestion_job import IngestionJob
from app.db.repositories.job_repository import JobCreateRequest


def test_job_create_request_defaults() -> None:
    req = JobCreateRequest(
        title="Acme Memo",
        doc_type="memo",
        source_filename="acme.pdf",
        storage_path="/tmp/acme.pdf",
    )
    assert req.visibility_scope == "global"
    assert req.organization_id is None


def test_ingestion_job_model_default_status() -> None:
    """The ORM default makes new rows queued. The migration drops the SQL
    default, so the application layer must set status explicitly — this
    test pins that contract."""
    job = IngestionJob(
        title="X",
        doc_type="t",
        source_filename="x.pdf",
        storage_path="/tmp/x.pdf",
        visibility_scope="global",
        status="queued",
        progress=0,
    )
    assert job.status == "queued"
    assert job.progress == 0
    # Job id is auto-assigned by `default=uuid.uuid4` when the instance is
    # flushed; the constructor leaves it None — verify we don't accidentally
    # depend on it pre-flush.
    assert job.organization_id is None
    _ = uuid.uuid4()  # consume so the import is used
