from pathlib import Path

from app.healthcare.fhir_parser import FHIRParseError, parse_fhir_file
from app.schemas.patient import PatientContext


class PatientNotFoundError(LookupError):
    pass


class PatientService:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def list_patients(self) -> list[PatientContext]:
        if not self.data_dir.exists():
            return []

        patients: list[PatientContext] = []

        for path in sorted(self.data_dir.glob("patient-*.json")):
            try:
                patients.append(parse_fhir_file(path))
            except FHIRParseError:
                continue

        return patients

    def get_patient(self, patient_id: str) -> PatientContext:
        safe_patient_id = patient_id.strip()

        if not safe_patient_id:
            raise PatientNotFoundError("Patient not found.")

        path = self.data_dir / f"{safe_patient_id}.json"

        if not path.exists():
            raise PatientNotFoundError(
                f"Patient '{safe_patient_id}' was not found."
            )

        try:
            return parse_fhir_file(path)
        except FHIRParseError as exc:
            raise PatientNotFoundError(
                f"Patient '{safe_patient_id}' could not be loaded."
            ) from exc


patient_service = PatientService(
    data_dir="/app/data/fhir/patients"
)
