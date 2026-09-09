from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ClinicalDocumentCategory(StrEnum):
    HYPERTENSION = "hypertension"
    DIABETES = "diabetes"
    CARDIOVASCULAR_RISK = "cardiovascular_risk"
    MEDICATION = "medication"
    PREVENTION = "prevention"
    LABORATORY = "laboratory"
    GENERAL = "general"


class ClinicalDocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    source_url: HttpUrl
    publication_date: date | None = None
    access_date: date
    category: ClinicalDocumentCategory
    version: str | None = None
    license_or_usage_note: str | None = None


class ClinicalSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    text: str = Field(min_length=1)
    order: int = Field(ge=0)


class ClinicalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: ClinicalDocumentMetadata
    sections: list[ClinicalSection] = Field(default_factory=list)
    raw_text: str | None = None


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    source_url: HttpUrl
    category: ClinicalDocumentCategory
    version: str | None = None
    publication_date: date | None = None
    access_date: date
    section_id: str | None = None
    section_heading: str | None = None
    chunk_number: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_count: int | None = Field(default=None, ge=0)
