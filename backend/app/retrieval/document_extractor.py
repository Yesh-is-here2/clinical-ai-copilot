from app.retrieval.html_extractor import (
    GuidelineExtractionError,
    extract_html_sections,
)
from app.retrieval.pdf_extractor import (
    PDFExtractionError,
    extract_pdf_sections,
)
from app.schemas.document import ClinicalSection
from app.schemas.ingestion import DownloadedGuideline


class DocumentExtractionError(ValueError):
    pass


def extract_document_sections(
    document: DownloadedGuideline,
) -> list[ClinicalSection]:
    content_type = (
        document.content_type
        .lower()
        .split(";", 1)[0]
        .strip()
    )

    try:
        if content_type in {
            "text/html",
            "application/xhtml+xml",
        }:
            return extract_html_sections(
                document.content
            )

        if content_type == "application/pdf":
            return extract_pdf_sections(
                document.content
            )

    except (
        GuidelineExtractionError,
        PDFExtractionError,
    ) as exc:
        raise DocumentExtractionError(
            f"Failed to extract document "
            f"'{document.document_id}': {exc}"
        ) from exc

    raise DocumentExtractionError(
        "Unsupported document content type: "
        f"'{document.content_type}'."
    )
