import re

from bs4 import BeautifulSoup, Tag

from app.schemas.document import ClinicalSection


class GuidelineExtractionError(ValueError):
    pass


REMOVE_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "button",
    "nav",
    "footer",
}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _section_text(
    heading: Tag,
) -> str:
    parts: list[str] = []

    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag):
            if sibling.name in {
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            }:
                break

            text = sibling.get_text(
                " ",
                strip=True,
            )

            if text:
                parts.append(text)

    return normalize_whitespace(
        "\n".join(parts)
    )


def extract_html_sections(
    content: bytes,
) -> list[ClinicalSection]:
    if not content:
        raise GuidelineExtractionError(
            "Cannot extract an empty HTML document."
        )

    soup = BeautifulSoup(
        content,
        "html.parser",
    )

    for tag_name in REMOVE_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    headings = main.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6"]
    )

    sections: list[ClinicalSection] = []

    for heading in headings:
        heading_text = normalize_whitespace(
            heading.get_text(
                " ",
                strip=True,
            )
        )

        if not heading_text:
            continue

        text = _section_text(heading)

        if not text:
            continue

        section_number = len(sections)

        sections.append(
            ClinicalSection(
                section_id=f"section-{section_number:04d}",
                heading=heading_text,
                text=text,
                order=section_number,
            )
        )

    if sections:
        return sections

    fallback_text = normalize_whitespace(
        main.get_text(
            "\n",
            strip=True,
        )
    )

    if not fallback_text:
        raise GuidelineExtractionError(
            "No usable text was found in the HTML document."
        )

    return [
        ClinicalSection(
            section_id="section-0000",
            heading="Document",
            text=fallback_text,
            order=0,
        )
    ]
