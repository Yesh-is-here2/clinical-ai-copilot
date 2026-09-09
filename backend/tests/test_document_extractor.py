import hashlib
import io
from datetime import datetime, timezone

import pytest
from reportlab.pdfgen import canvas

from app.retrieval.document_extractor import (
    DocumentExtractionError,
    extract_document_sections,
)
from app.schemas.ingestion import DownloadedGuideline


def make_downloaded_guideline(
    *,
    content: bytes,
    content_type: str,
) -> DownloadedGuideline:
    return DownloadedGuideline(
        document_id="test-guideline",
        source_url="https://example.org/guideline",
        final_url="https://example.org/guideline",
        content_type=content_type,
        content_length=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        downloaded_at=datetime.now(timezone.utc),
        content=content,
    )


def make_pdf(text: str) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)

    text_object = pdf.beginText(
        72,
        750,
    )

    for line in text.splitlines():
        text_object.textLine(line)

    pdf.drawText(text_object)
    pdf.showPage()
    pdf.save()

    return buffer.getvalue()


def test_dispatches_html_document() -> None:
    content = b"""
    <html>
      <body>
        <main>
          <h1>Clinical Guideline</h1>
          <p>Hypertension evidence.</p>
        </main>
      </body>
    </html>
    """

    document = make_downloaded_guideline(
        content=content,
        content_type="text/html",
    )

    sections = extract_document_sections(
        document
    )

    assert len(sections) == 1
    assert sections[0].heading == (
        "Clinical Guideline"
    )
    assert (
        "Hypertension evidence."
        in sections[0].text
    )


def test_dispatches_xhtml_document() -> None:
    content = b"""
    <html>
      <body>
        <main>
          <h1>Evidence</h1>
          <p>Clinical content.</p>
        </main>
      </body>
    </html>
    """

    document = make_downloaded_guideline(
        content=content,
        content_type=(
            "application/xhtml+xml; charset=utf-8"
        ),
    )

    sections = extract_document_sections(
        document
    )

    assert len(sections) == 1


def test_dispatches_pdf_document() -> None:
    content = make_pdf(
        "Clinical Guideline\n"
        "PDF clinical evidence."
    )

    document = make_downloaded_guideline(
        content=content,
        content_type="application/pdf",
    )

    sections = extract_document_sections(
        document
    )

    assert len(sections) == 1
    assert sections[0].heading == "Page 1"
    assert (
        "PDF clinical evidence."
        in sections[0].text
    )


def test_rejects_unsupported_content_type() -> None:
    content = b"unsupported"

    document = make_downloaded_guideline(
        content=content,
        content_type="text/plain",
    )

    with pytest.raises(
        DocumentExtractionError,
        match="Unsupported document content type",
    ):
        extract_document_sections(
            document
        )


def test_wraps_html_extraction_failure() -> None:
    content = b"""
    <html>
      <body>
        <script>
          no usable evidence
        </script>
      </body>
    </html>
    """

    document = make_downloaded_guideline(
        content=content,
        content_type="text/html",
    )

    with pytest.raises(
        DocumentExtractionError,
        match="test-guideline",
    ):
        extract_document_sections(
            document
        )


def test_wraps_pdf_extraction_failure() -> None:
    content = b"not-a-valid-pdf"

    document = make_downloaded_guideline(
        content=content,
        content_type="application/pdf",
    )

    with pytest.raises(
        DocumentExtractionError,
        match="test-guideline",
    ):
        extract_document_sections(
            document
        )
