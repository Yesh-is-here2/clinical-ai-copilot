import io

import pytest
from reportlab.pdfgen import canvas

from app.retrieval.pdf_extractor import (
    PDFExtractionError,
    extract_pdf_sections,
    normalize_pdf_text,
)


def make_pdf(
    pages: list[str],
) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)

    for page_text in pages:
        text_object = pdf.beginText(72, 750)

        for line in page_text.splitlines():
            text_object.textLine(line)

        pdf.drawText(text_object)
        pdf.showPage()

    pdf.save()

    return buffer.getvalue()


def test_normalize_pdf_text() -> None:
    text = (
        "Blood   pressure\n"
        "manage-\n"
        "ment\n\n\n"
        "Follow-up"
    )

    result = normalize_pdf_text(text)

    assert result == (
        "Blood pressure\n"
        "management\n\n"
        "Follow-up"
    )


def test_extract_single_page_pdf() -> None:
    content = make_pdf(
        [
            "Hypertension Guideline\n"
            "Clinical evidence for blood pressure."
        ]
    )

    sections = extract_pdf_sections(content)

    assert len(sections) == 1
    assert sections[0].section_id == "page-0001"
    assert sections[0].heading == "Page 1"
    assert "Hypertension Guideline" in sections[0].text
    assert (
        "Clinical evidence for blood pressure."
        in sections[0].text
    )


def test_extract_multiple_pdf_pages() -> None:
    content = make_pdf(
        [
            "Page one clinical evidence.",
            "Page two clinical evidence.",
            "Page three clinical evidence.",
        ]
    )

    sections = extract_pdf_sections(content)

    assert len(sections) == 3

    assert sections[0].section_id == "page-0001"
    assert sections[1].section_id == "page-0002"
    assert sections[2].section_id == "page-0003"

    assert sections[0].order == 0
    assert sections[1].order == 1
    assert sections[2].order == 2

    assert (
        "Page two clinical evidence."
        in sections[1].text
    )


def test_pdf_section_ids_are_deterministic() -> None:
    content = make_pdf(
        [
            "First page.",
            "Second page.",
        ]
    )

    first = extract_pdf_sections(content)
    second = extract_pdf_sections(content)

    assert [
        section.section_id
        for section in first
    ] == [
        section.section_id
        for section in second
    ]


def test_pdf_extractor_rejects_empty_content() -> None:
    with pytest.raises(
        PDFExtractionError,
        match="empty PDF",
    ):
        extract_pdf_sections(b"")


def test_pdf_extractor_rejects_invalid_pdf() -> None:
    with pytest.raises(
        PDFExtractionError,
        match="could not be parsed",
    ):
        extract_pdf_sections(
            b"This is not a PDF."
        )


def test_pdf_extractor_rejects_pdf_without_text() -> None:
    content = make_pdf([""])

    with pytest.raises(
        PDFExtractionError,
        match="No extractable text",
    ):
        extract_pdf_sections(content)
