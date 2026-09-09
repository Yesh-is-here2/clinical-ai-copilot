from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.schemas.document import ClinicalDocument


class IngestionStatus(StrEnum):
    DOWNLOADED = "downloaded"
    PROCESSED = "processed"
    FAILED = "failed"


class DownloadedGuideline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    source_url: HttpUrl
    final_url: HttpUrl
    content_type: str = Field(min_length=1)
    content_length: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    downloaded_at: datetime
    status: IngestionStatus = IngestionStatus.DOWNLOADED
    content: bytes


class IngestionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    source_url: HttpUrl
    final_url: HttpUrl
    content_type: str = Field(min_length=1)
    content_length: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    downloaded_at: datetime
    processed_at: datetime
    section_count: int = Field(gt=0)


class IngestedGuideline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: IngestionStatus = IngestionStatus.PROCESSED
    document: ClinicalDocument
    provenance: IngestionProvenance
