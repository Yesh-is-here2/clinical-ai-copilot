import io
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.schemas.document import ClinicalSection


class PDFExtractionError(ValueError):
    pass


def normalize_pdf_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_sections(
    content: bytes,
) -> list[ClinicalSection]:
    if not content:
        raise PDFExtractionError(
            "Cannot extract an empty PDF document."
        )

    try:
        reader = PdfReader(io.BytesIO(content))
    except (PdfReadError, ValueError, OSError) as exc:
        raise PDFExtractionError(
            "The PDF document could not be parsed."
        ) from exc

    if reader.is_encrypted:
        try:
            result = reader.decrypt("")
        except Exception as exc:
            raise PDFExtractionError(
                "Encrypted PDF documents are not supported."
            ) from exc

        if result == 0:
            raise PDFExtractionError(
                "Encrypted PDF documents are not supported."
            )

    if not reader.pages:
        raise PDFExtractionError(
            "The PDF document contains no pages."
        )

    sections: list[ClinicalSection] = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise PDFExtractionError(
                f"Text extraction failed on PDF page {page_number}."
            ) from exc

        text = normalize_pdf_text(raw_text)

        if not text:
            continue

        section_number = len(sections)

        sections.append(
            ClinicalSection(
                section_id=f"page-{page_number:04d}",
                heading=f"Page {page_number}",
                text=text,
                order=section_number,
            )
        )

    if not sections:
        raise PDFExtractionError(
            "No extractable text was found in the PDF document. "
            "Scanned or image-only PDFs require OCR and are not "
            "supported by this ingestion path."
        )

    return sections
