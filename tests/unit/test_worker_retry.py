"""Worker retry contract (``app.jobs.worker.ingest_pdf_task``).

Arq only re-queues a task that raises ``arq.worker.Retry``. The worker
used to re-raise the original exception instead, which Arq records as a
final failure — the job row was left at ``processing`` forever with no
error message, and the upload UI spun on "Reading…" indefinitely (seen
in production when the OpenAI key was revoked and every embed call
failed). These tests pin both halves: retry while attempts remain, then
a ``failed`` row carrying the real error.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from arq.worker import Retry

from app.core.config import get_settings
from app.jobs import worker


class _FakeSession:
    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeJobRepo:
    """Shared state across the worker's several sessions, like the DB row."""

    def __init__(self, job: SimpleNamespace) -> None:
        self.job = job

    def __call__(self, _db: object) -> _FakeJobRepo:
        return self

    def get(self, _job_id: uuid.UUID) -> SimpleNamespace:
        return self.job

    def mark_processing(self, _job_id: uuid.UUID) -> None:
        self.job.status = "processing"

    def bump_retry(self, _job_id: uuid.UUID) -> int:
        self.job.retry_count += 1
        return self.job.retry_count

    def mark_failed(self, _job_id: uuid.UUID, *, error: str) -> None:
        self.job.status = "failed"
        self.job.error_message = error


class _FailingIngestion:
    def __init__(self, _db: object) -> None:
        pass

    def ingest_pdf(self, **_kwargs: Any) -> None:
        raise RuntimeError("Incorrect API key provided")


@pytest.fixture()
def job_repo(monkeypatch: pytest.MonkeyPatch) -> _FakeJobRepo:
    job = SimpleNamespace(
        status="queued",
        retry_count=0,
        error_message=None,
        document_id=None,
        visibility_scope="global",
        document_origin=None,
        organization_id=None,
        user_id=None,
        source_filename="a.pdf",
        title="A",
        doc_type="auto",
        storage_path="/nonexistent",
    )
    repo = _FakeJobRepo(job)
    monkeypatch.setattr(worker, "SessionLocal", _FakeSession)
    monkeypatch.setattr(worker, "JobRepository", repo)
    monkeypatch.setattr(worker, "IngestionService", _FailingIngestion)
    return repo


async def test_failure_with_attempts_left_raises_arq_retry(job_repo: _FakeJobRepo) -> None:
    with pytest.raises(Retry):
        await worker.ingest_pdf_task({}, str(uuid.uuid4()), b"%PDF")
    assert job_repo.job.retry_count == 1
    assert job_repo.job.status == "processing"


async def test_out_of_attempts_marks_job_failed_with_error(job_repo: _FakeJobRepo) -> None:
    job_repo.job.retry_count = get_settings().ingestion_max_retries

    result = await worker.ingest_pdf_task({}, str(uuid.uuid4()), b"%PDF")

    assert result["status"] == "failed"
    assert job_repo.job.status == "failed"
    assert "Incorrect API key" in job_repo.job.error_message
