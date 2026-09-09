from datetime import date

from app.retrieval.sources import (
    GUIDELINE_SOURCES,
    get_guideline_source,
    list_enabled_guideline_sources,
)


def test_guideline_registry_has_unique_ids() -> None:
    document_ids = [
        source.document_id
        for source in GUIDELINE_SOURCES
    ]

    assert len(document_ids) == len(set(document_ids))


def test_registry_contains_expected_sources() -> None:
    document_ids = {
        source.document_id
        for source in GUIDELINE_SOURCES
    }

    assert "nice-ng136-hypertension" in document_ids
    assert "who-hypertension-pharmacological-2021" in document_ids
    assert "cdc-high-blood-pressure-about" in document_ids


def test_get_guideline_source() -> None:
    source = get_guideline_source(
        "nice-ng136-hypertension"
    )

    assert source is not None
    assert source.organization == (
        "National Institute for Health and Care Excellence"
    )


def test_unknown_guideline_source_returns_none() -> None:
    assert get_guideline_source("unknown-source") is None


def test_enabled_guidelines() -> None:
    enabled_sources = list_enabled_guideline_sources()

    assert enabled_sources
    assert all(source.enabled for source in enabled_sources)



def test_who_publication_date() -> None:
    source = get_guideline_source(
        "who-hypertension-pharmacological-2021"
    )

    assert source is not None
    assert source.publication_date == date(
        2021,
        8,
        24,
    )


def test_who_source_uses_current_pdf_content_endpoint() -> None:
    source = get_guideline_source(
        "who-hypertension-pharmacological-2021"
    )

    assert source is not None
    assert str(source.source_url) == (
        "https://iris.who.int/server/api/core/bitstreams/"
        "f062769d-f075-4a00-87af-0a2106e0bd04/content"
    )
    assert source.allowed_hosts == (
        "iris.who.int",
    )
    assert source.ingestion_method == "pdf"

