from datetime import date

from pydantic import BaseModel, ConfigDict, HttpUrl

from app.schemas.document import ClinicalDocumentCategory


class GuidelineSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    title: str
    organization: str
    source_url: HttpUrl
    publication_date: date | None
    category: ClinicalDocumentCategory
    version: str | None = None
    license_or_usage_note: str
    ingestion_method: str
    allowed_hosts: tuple[str, ...]
    enabled: bool = True


GUIDELINE_SOURCES: tuple[GuidelineSource, ...] = (
    GuidelineSource(
        document_id="nice-ng136-hypertension",
        title="Hypertension in adults: diagnosis and management",
        organization="National Institute for Health and Care Excellence",
        source_url="https://www.nice.org.uk/guidance/ng136",
        publication_date=date(2019, 8, 28),
        category=ClinicalDocumentCategory.HYPERTENSION,
        version="NG136; updated 2026-02-26",
        license_or_usage_note=(
            "Public NICE guidance. Preserve attribution and source URL. "
            "Verify reuse requirements before redistribution of substantial source text."
        ),
        ingestion_method="html",
        allowed_hosts=("www.nice.org.uk", "nice.org.uk"),
    ),
    GuidelineSource(
        document_id="who-hypertension-pharmacological-2021",
        title="Guideline for the pharmacological treatment of hypertension in adults",
        organization="World Health Organization",
        source_url=(
            "https://iris.who.int/server/api/core/bitstreams/"
            "f062769d-f075-4a00-87af-0a2106e0bd04/content"
        ),
        publication_date=date(2021, 8, 24),
        category=ClinicalDocumentCategory.HYPERTENSION,
        version="2021",
        license_or_usage_note=(
            "WHO publication distributed under CC BY-NC-SA 3.0 IGO. "
            "Attribution and applicable licence conditions must be preserved."
        ),
        ingestion_method="pdf",
        allowed_hosts=("iris.who.int",),
    ),
    GuidelineSource(
        document_id="cdc-high-blood-pressure-about",
        title="About High Blood Pressure",
        organization="Centers for Disease Control and Prevention",
        source_url="https://www.cdc.gov/high-blood-pressure/about/index.html",
        publication_date=date(2026, 1, 28),
        category=ClinicalDocumentCategory.HYPERTENSION,
        version="2026-01-28",
        license_or_usage_note=(
            "Public CDC informational source. Preserve attribution and source URL. "
            "Do not treat this registry entry as an assertion that all CDC content is unrestricted."
        ),
        ingestion_method="html",
        allowed_hosts=("www.cdc.gov", "cdc.gov"),
    ),
)


def get_guideline_source(document_id: str) -> GuidelineSource | None:
    normalized_id = document_id.strip()

    if not normalized_id:
        return None

    for source in GUIDELINE_SOURCES:
        if source.document_id == normalized_id:
            return source

    return None


def list_enabled_guideline_sources() -> list[GuidelineSource]:
    return [
        source
        for source in GUIDELINE_SOURCES
        if source.enabled
    ]




