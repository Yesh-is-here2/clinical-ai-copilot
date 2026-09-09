import pytest

from app.retrieval.html_extractor import (
    GuidelineExtractionError,
    extract_html_sections,
    normalize_whitespace,
)


def test_normalize_whitespace() -> None:
    result = normalize_whitespace(
        "  Blood   pressure \n\n\n management  "
    )

    assert result == "Blood pressure\n\nmanagement"


def test_extract_html_sections() -> None:
    content = b"""
    <html>
      <body>
        <nav>Navigation should disappear</nav>

        <main>
          <h1>Hypertension Guideline</h1>
          <p>Guideline overview.</p>

          <h2>Diagnosis</h2>
          <p>Diagnosis evidence.</p>
          <ul>
            <li>Confirm blood pressure measurements.</li>
            <li>Review relevant clinical context.</li>
          </ul>

          <h2>Management</h2>
          <p>Management evidence.</p>
        </main>

        <footer>Footer should disappear</footer>
      </body>
    </html>
    """

    sections = extract_html_sections(content)

    assert len(sections) == 3

    assert sections[0].heading == (
        "Hypertension Guideline"
    )
    assert sections[0].text == (
        "Guideline overview."
    )

    assert sections[1].heading == "Diagnosis"
    assert "Diagnosis evidence." in sections[1].text
    assert (
        "Confirm blood pressure measurements."
        in sections[1].text
    )

    assert sections[2].heading == "Management"
    assert sections[2].order == 2


def test_html_extractor_removes_script_and_navigation() -> None:
    content = b"""
    <html>
      <body>
        <nav>Ignore navigation</nav>
        <main>
          <h1>Clinical Evidence</h1>
          <p>Keep this evidence.</p>
          <script>
            Ignore this script content.
          </script>
        </main>
      </body>
    </html>
    """

    sections = extract_html_sections(content)

    combined = " ".join(
        section.text
        for section in sections
    )

    assert "Keep this evidence." in combined
    assert "Ignore navigation" not in combined
    assert "Ignore this script content" not in combined


def test_html_extractor_uses_fallback_without_headings() -> None:
    content = b"""
    <html>
      <body>
        <main>
          <p>Clinical evidence without headings.</p>
        </main>
      </body>
    </html>
    """

    sections = extract_html_sections(content)

    assert len(sections) == 1
    assert sections[0].heading == "Document"
    assert (
        sections[0].text
        == "Clinical evidence without headings."
    )


def test_html_extractor_rejects_empty_content() -> None:
    with pytest.raises(
        GuidelineExtractionError,
        match="empty HTML",
    ):
        extract_html_sections(b"")


def test_html_extractor_rejects_document_without_text() -> None:
    content = b"""
    <html>
      <body>
        <script>
          console.log("nothing useful");
        </script>
      </body>
    </html>
    """

    with pytest.raises(
        GuidelineExtractionError,
        match="No usable text",
    ):
        extract_html_sections(content)


def test_section_ids_are_deterministic() -> None:
    content = b"""
    <main>
      <h1>Overview</h1>
      <p>Overview text.</p>

      <h2>Diagnosis</h2>
      <p>Diagnosis text.</p>
    </main>
    """

    first = extract_html_sections(content)
    second = extract_html_sections(content)

    assert [
        section.section_id
        for section in first
    ] == [
        section.section_id
        for section in second
    ]

    assert first[0].section_id == "section-0000"
    assert first[1].section_id == "section-0001"
