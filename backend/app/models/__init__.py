"""Import every ORM model so ``Base.metadata`` is fully populated.

Alembic's ``env.py`` imports this package (not the individual model
modules) before comparing/generating against ``Base.metadata`` - if a
model module isn't imported somewhere, SQLAlchemy never learns its table
exists, and Alembic autogenerate would silently omit it. Import this
package, not a single model, whenever you need "all tables".
"""

from app.db.base import Base
from app.models.audit import AuditLogRecord
from app.models.document import (
    ClinicalDocumentRecord,
    DocumentChunkRecord,
    DocumentEmbeddingRecord,
)
from app.models.evaluation import EvaluationResultRecord
from app.models.patient import FHIRResource, Patient
from app.models.query import (
    CitationRecord,
    GeneratedResponseRecord,
    QueryRecord,
    RetrievalResultRecord,
)
from app.models.user import User, UserRole

__all__ = [
    "Base",
    "User",
    "UserRole",
    "Patient",
    "FHIRResource",
    "ClinicalDocumentRecord",
    "DocumentChunkRecord",
    "DocumentEmbeddingRecord",
    "QueryRecord",
    "RetrievalResultRecord",
    "GeneratedResponseRecord",
    "CitationRecord",
    "EvaluationResultRecord",
    "AuditLogRecord",
]
