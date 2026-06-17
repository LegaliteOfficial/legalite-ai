"""Legacy document upload — preserved for the current mobile/web client.

Both the original ``POST /upload-law`` and the new tenant-aware uploads share
``IngestionService.ingest_pdf``. The legacy route maps to GLOBAL visibility
unconditionally (matching pre-tenancy behavior); ``POST /admin/upload-law``
lives in ``admin.py``, ``POST /documents/private-upload`` in
``private_documents.py``.

The bulk endpoint ``POST /upload-law/bulk`` lives in this file too — it
follows the legacy route's anonymous + global contract but enqueues each
PDF through the Arq worker so a 25-file batch returns in milliseconds
instead of blocking on serial inline ingestion.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.api.dependencies.services import get_ingestion_service
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.enums import ConfidentialityLevel, VisibilityScope
from app.db.session import get_db
from app.schemas.documents import UploadLawResponse
from app.schemas.jobs import (
    BulkUploadJobResult,
    BulkUploadRejection,
    BulkUploadResponse,
)
from app.services.ingestion_service import IngestionService
from app.services.upload_dispatcher import dispatch_upload

logger = get_logger(__name__)

router = APIRouter(prefix="", tags=["documents"])

_TITLE_MIN_LENGTH = 2
_TITLE_MAX_LENGTH = 512
_WHITESPACE_RE = re.compile(r"\s+")


@router.post(
    "/upload-law",
    response_model=UploadLawResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_law(
    file: UploadFile = File(..., description="Ghana legal PDF"),
    title: str = Form(..., min_length=2, max_length=512),
    doc_type: str = Form(..., description="Constitution | Act | Regulation | Case Law | ..."),
    service: IngestionService = Depends(get_ingestion_service),
) -> UploadLawResponse:
    """Legacy ingestion endpoint — always GLOBAL.

    Kept for backward compatibility with the current mobile/web client.
    New code should call ``/admin/upload-law`` (GLOBAL, admin-gated) or
    ``/documents/private-upload`` (TENANT_PRIVATE, requires org context).
    """
    settings = get_settings()

    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_mb}MB limit",
        )

    try:
        document_id, chunks_indexed = service.ingest_pdf(
            file_bytes=content,
            original_filename=file.filename,
            title=title,
            doc_type=doc_type,
            visibility_scope=VisibilityScope.GLOBAL,
        )
    except Exception as exc:  # noqa: BLE001 — surface a clean 500 to the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return UploadLawResponse(
        document_id=document_id,
        title=title,
        doc_type=doc_type,
        page_count=0,
        chunks_indexed=chunks_indexed,
        status="ingested" if chunks_indexed > 0 else "failed",
    )


@router.post(
    "/upload-law/bulk",
    response_model=BulkUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Bulk-upload multiple Ghana legal PDFs in a single request",
)
async def upload_law_bulk(
    response: Response,
    files: list[UploadFile] = File(
        ...,
        description=(
            "Up to ``BULK_UPLOAD_MAX_FILES`` PDFs (default 25). Each "
            "file's title is derived from its filename minus the .pdf "
            "extension."
        ),
    ),
    doc_type: str = Form(
        ...,
        description=(
            "Doc type applied to every file in the batch. Mix doc types "
            "across separate calls — one batch, one doc type."
        ),
    ),
    db: Session = Depends(get_db),
) -> BulkUploadResponse:
    """Enqueue many PDFs at once. Legacy contract: anonymous, GLOBAL scope.

    Why a separate endpoint instead of looping ``/upload-law``:
    - One HTTP round-trip for the whole batch instead of N.
    - Per-file ingestion runs on the Arq worker with built-in retries
      (``INGESTION_MAX_RETRIES``), so transient OpenAI / Qdrant
      blips don't kill the upload.
    - Per-file failures (wrong type, too large, name collision) are
      reported alongside successes — the rest of the batch still
      goes through. A single bad apple never wastes the whole upload.
    - Title derivation is server-side and idempotent: clients don't
      have to thread N titles through the multipart form, just send
      meaningfully-named PDFs.

    The endpoint returns **immediately** after the last enqueue; actual
    ingestion happens in the worker pool. Poll
    ``GET /jobs/{job_id}`` for each returned ``job_id`` to track
    progress.

    The same dispatcher fallback as the admin path applies: if the
    Arq queue is unreachable (Redis down) the file ingests inline and
    is reported as ``status: completed`` instead of ``queued``.
    """
    settings = get_settings()

    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No files were uploaded.",
        )
    if len(files) > settings.bulk_upload_max_files:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Batch contains {len(files)} files; the per-request "
                f"cap is {settings.bulk_upload_max_files}. Split the "
                "upload into smaller batches."
            ),
        )

    doc_type_clean = doc_type.strip()
    if len(doc_type_clean) < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="doc_type is required.",
        )

    accepted: list[BulkUploadJobResult] = []
    rejected: list[BulkUploadRejection] = []

    for upload in files:
        filename = upload.filename or "<unnamed>"
        try:
            content, title = await _validate_and_extract_title(
                upload, max_upload_mb=settings.max_upload_mb
            )
        except _BulkRejected as exc:
            rejected.append(BulkUploadRejection(filename=filename, reason=exc.reason))
            continue

        # Each ``dispatch_upload`` commits its own transaction (the
        # job row + storage write are durable BEFORE the enqueue
        # call). Per-file commits keep one bad file from invalidating
        # the rest of the batch.
        try:
            job = await dispatch_upload(
                db,
                file_bytes=content,
                original_filename=filename,
                title=title,
                doc_type=doc_type_clean,
                visibility_scope=VisibilityScope.GLOBAL,
                document_origin=None,
                confidentiality_level=ConfidentialityLevel.PUBLIC,
                tenant=None,
            )
        except Exception as exc:  # noqa: BLE001
            # The dispatcher already absorbs Redis failures (falls back
            # to inline). Anything reaching here is a hard failure —
            # corrupt PDF rejected by the parser before enqueue, disk
            # full, etc. Log it and surface to the caller per-file.
            logger.exception("bulk_upload.file_failed", filename=filename)
            rejected.append(
                BulkUploadRejection(
                    filename=filename,
                    reason=f"Server-side enqueue failed: {exc}",
                )
            )
            continue

        accepted.append(
            BulkUploadJobResult(
                filename=filename,
                title=title,
                job_id=job.job_id,
                status=job.status,
                document_id=job.document_id,
            )
        )

    logger.info(
        "bulk_upload.completed",
        total=len(files),
        accepted=len(accepted),
        rejected=len(rejected),
    )

    # HTTP semantics: if the whole batch was rejected, flip to 422
    # so the client sees a hard failure status. Mixed batches stay
    # at 202 — at least one file is being processed.
    if not accepted:
        response.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    return BulkUploadResponse(
        accepted=len(accepted),
        rejected=len(rejected),
        max_files_per_request=settings.bulk_upload_max_files,
        jobs=accepted,
        rejections=rejected,
    )


# ── Helpers ────────────────────────────────────────────────────────────────


class _BulkRejected(Exception):
    """Internal sentinel — a single file failed validation cleanly.

    Carries the user-facing reason so the route handler can append a
    BulkUploadRejection without leaking traceback noise.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def _validate_and_extract_title(
    upload: UploadFile, *, max_upload_mb: int
) -> tuple[bytes, str]:
    """Read + validate one file. Returns (bytes, derived_title).

    Title derivation:
      1. Strip everything from the last ``.pdf`` onward (case-insensitive).
      2. Collapse runs of whitespace to a single space.
      3. Trim trailing/leading punctuation noise (``_-.`` etc.) so a
         filename like ``__Acquah_v_Kwesi__.pdf`` becomes
         ``Acquah_v_Kwesi`` not ``__Acquah_v_Kwesi__``.
      4. Clamp to ``_TITLE_MAX_LENGTH``.

    Raises ``_BulkRejected`` (with a user-facing reason) when the file
    fails validation. Reasons are deliberately short — they surface in
    the bulk response and should fit a UI toast.
    """
    if not upload.filename:
        raise _BulkRejected("Missing filename in upload.")
    if Path(upload.filename).suffix.lower() != ".pdf":
        raise _BulkRejected("Only PDF files are accepted.")

    content = await upload.read()
    if not content:
        raise _BulkRejected("File is empty.")
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_upload_mb:
        raise _BulkRejected(
            f"File exceeds {max_upload_mb} MB limit ({size_mb:.1f} MB)."
        )

    title = _filename_to_title(upload.filename)
    if len(title) < _TITLE_MIN_LENGTH:
        raise _BulkRejected(
            f"Derived title too short (need ≥ {_TITLE_MIN_LENGTH} chars). "
            "Rename the file to something meaningful."
        )
    return content, title


def _filename_to_title(filename: str) -> str:
    """Strip the .pdf extension and normalise whitespace."""
    # ``Path.stem`` only strips the LAST extension, so ``foo.pdf`` →
    # ``foo`` and ``Republic v. X (2024).pdf`` → ``Republic v. X (2024)``.
    base = Path(filename).stem
    base = _WHITESPACE_RE.sub(" ", base).strip()
    # Strip a small set of punctuation that frequently appears at the
    # edges of exported filenames but reads as noise in a citation:
    # underscores, hyphens, dots, parentheses-spaces. We do NOT strip
    # them globally — they're meaningful inside the title.
    base = base.strip("._- ")
    return base[:_TITLE_MAX_LENGTH]
