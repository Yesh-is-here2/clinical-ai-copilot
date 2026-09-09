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


def test_html_extractor_uses_article_when_no_main_tag() -> None:
    # Some guideline sites (e.g. content-management-system-rendered pages)
    # wrap the guidance body in <article> rather than <main>. This is a
    # regression fixture for that DOM pattern, not scraped real content.
    content = b"""
    <html>
      <body>
        <nav>Breadcrumb: Home &gt; Guidance &gt; Hypertension</nav>

        <article>
          <h1>Hypertension in adults</h1>
          <p>Diagnosis and management guidance.</p>

          <h2>Recommendations</h2>
          <p>Confirm the diagnosis before starting treatment.</p>
        </article>

        <footer>Crown copyright</footer>
      </body>
    </html>
    """

    sections = extract_html_sections(content)
    combined = " ".join(section.text for section in sections)

    assert len(sections) == 2
    assert sections[0].heading == "Hypertension in adults"
    assert "Breadcrumb" not in combined
    assert "Crown copyright" not in combined


def test_html_extractor_regression_nice_style_page() -> None:
    # A synthetic fixture modeled on the general shape of a NICE-style
    # guidance page: cookie banner and related-content sidebar outside
    # the article, real guidance content with nested headings inside it.
    content = b"""
    <html>
      <body>
        <div id="cookie-banner">We use cookies. Accept?</div>
        <nav id="breadcrumb">Home &gt; Guidance &gt; NG136</nav>

        <article>
          <h1>Hypertension in adults: diagnosis and management</h1>
          <p>This guideline covers identifying and treating hypertension.</p>

          <h2>1 Recommendations</h2>
          <p>Offer lifestyle advice alongside pharmacological treatment.</p>

          <h3>1.1 Diagnosing hypertension</h3>
          <p>Use ambulatory blood pressure monitoring to confirm diagnosis.</p>
        </article>

        <aside id="related-content">
          <h2>Related NICE guidance</h2>
          <a href="/guidance/ng28">Type 2 diabetes</a>
        </aside>

        <footer>&copy; NICE. All rights reserved.</footer>
      </body>
    </html>
    """

    sections = extract_html_sections(content)
    headings = [section.heading for section in sections]

    assert "Hypertension in adults: diagnosis and management" in headings
    assert "1 Recommendations" in headings
    assert "1.1 Diagnosing hypertension" in headings

    combined = " ".join(section.text for section in sections)
    assert "cookies" not in combined.lower()
    assert "All rights reserved" not in combined


def test_html_extractor_regression_cdc_style_page() -> None:
    # A synthetic fixture modeled on the general shape of a CDC
    # informational page: global header/nav, a "page last reviewed" date
    # bar, and content organized under h2 sections inside <main>.
    content = b"""
    <html>
      <body>
        <header>
          <nav>CDC.gov navigation</nav>
        </header>

        <main>
          <h1>About High Blood Pressure</h1>
          <p>High blood pressure has no symptoms for most people.</p>

          <h2>Health Effects</h2>
          <p>Uncontrolled blood pressure can damage the heart and kidneys.</p>

          <h2>Prevention</h2>
          <p>Maintain a healthy weight and limit sodium intake.</p>
        </main>

        <div id="last-reviewed">Page last reviewed: January 28, 2026</div>
        <footer>
          <nav>CDC.gov footer links</nav>
        </footer>
      </body>
    </html>
    """

    sections = extract_html_sections(content)
    headings = [section.heading for section in sections]

    assert headings == [
        "About High Blood Pressure",
        "Health Effects",
        "Prevention",
    ]

    combined = " ".join(section.text for section in sections)
    assert "CDC.gov navigation" not in combined
    assert "CDC.gov footer links" not in combined


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
