import uuid
from datetime import date

from sqlalchemy import Boolean, Date, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin


class Patient(TimestampMixin, Base):
    """Persisted counterpart of the file-backed synthetic FHIR patients.

    ``patient_id`` deliberately mirrors the existing file-backed IDs
    (``"patient-001"`` etc, see ``app.services.patient.PatientService``)
    rather than a surrogate UUID, so this table can be populated from - and
    stay in sync with - ``data/fhir/patients/*.json`` without an ID
    translation layer. The Phase 2 file-backed API is unaffected: this
    table only exists to support features that need patients addressable
    from SQL (queries, audit logs), it does not replace
    ``GET /patients``/``GET /patients/{id}``.
    """

    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    sex: Mapped[str | None] = mapped_column(String(32), nullable=True)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    synthetic: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )

    fhir_resources: Mapped[list["FHIRResource"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"Patient(patient_id={self.patient_id!r})"


class FHIRResource(TimestampMixin, Base):
    """One raw FHIR resource entry (Patient/Observation/Condition/...)
    belonging to a synthetic patient bundle, stored verbatim as JSONB for
    provenance/auditability alongside the normalized ``PatientContext``
    the API actually serves."""

    __tablename__ = "fhir_resources"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("patients.patient_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(128), nullable=False)
    raw_resource: Mapped[dict] = mapped_column(JSONB, nullable=False)

    patient: Mapped["Patient"] = relationship(back_populates="fhir_resources")

    __table_args__ = (
        UniqueConstraint(
            "patient_id",
            "resource_type",
            "resource_id",
            name="uq_fhir_resource_identity",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            f"FHIRResource(patient_id={self.patient_id!r}, "
            f"resource_type={self.resource_type!r}, "
            f"resource_id={self.resource_id!r})"
        )
