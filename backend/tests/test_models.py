"""ORM model metadata / relationship / constraint tests.

None of these touch a database - they only inspect
``Base.metadata``/mapper configuration, which is exactly what lets this
suite run without PostgreSQL (see backend/pytest.ini).
"""

from sqlalchemy.orm import configure_mappers

from app.core.constants import DEFAULT_EMBEDDING_DIMENSIONS
from app.models import Base


EXPECTED_TABLES = {
    "users",
    "patients",
    "fhir_resources",
    "clinical_documents",
    "document_chunks",
    "document_embeddings",
    "queries",
    "retrieval_results",
    "generated_responses",
    "citations",
    "evaluation_results",
    "audit_logs",
}


def test_all_required_tables_are_registered() -> None:
    assert set(Base.metadata.tables.keys()) == EXPECTED_TABLES


def test_mappers_configure_without_error() -> None:
    # Raises if any relationship() is misconfigured (bad back_populates,
    # dangling FK, etc.) - this is the cheapest possible check that the
    # whole model graph is internally consistent.
    configure_mappers()


def test_every_table_has_a_primary_key() -> None:
    for name, table in Base.metadata.tables.items():
        assert table.primary_key.columns.keys(), f"{name} has no primary key"


def test_every_table_has_created_and_updated_at() -> None:
    for name, table in Base.metadata.tables.items():
        assert "created_at" in table.columns, f"{name} missing created_at"
        assert "updated_at" in table.columns, f"{name} missing updated_at"


def test_foreign_keys_point_at_real_tables_and_columns() -> None:
    for table in Base.metadata.tables.values():
        for fk in table.foreign_keys:
            target_table, target_column = fk.target_fullname.split(".")
            assert target_table in Base.metadata.tables
            assert target_column in Base.metadata.tables[target_table].columns


def test_document_chunks_has_unique_constraint_on_document_and_number() -> None:
    table = Base.metadata.tables["document_chunks"]
    unique_column_sets = {
        frozenset(c.name for c in constraint.columns)
        for constraint in table.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }

    assert frozenset({"document_id", "chunk_number"}) in unique_column_sets


def test_document_embeddings_has_unique_constraint_on_chunk_and_model() -> None:
    table = Base.metadata.tables["document_embeddings"]
    unique_column_sets = {
        frozenset(c.name for c in constraint.columns)
        for constraint in table.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }

    assert frozenset({"chunk_id", "model"}) in unique_column_sets


def test_embedding_column_dimension_matches_configured_constant() -> None:
    table = Base.metadata.tables["document_embeddings"]
    embedding_column = table.columns["embedding"]

    assert embedding_column.type.dim == DEFAULT_EMBEDDING_DIMENSIONS


def test_users_email_is_unique() -> None:
    table = Base.metadata.tables["users"]
    email_column = table.columns["email"]

    assert email_column.unique or any(
        frozenset({"email"})
        == frozenset(c.name for c in constraint.columns)
        for constraint in table.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    ) or any(
        list(index.columns) == [email_column] and index.unique
        for index in table.indexes
    )


def test_audit_logs_patient_id_is_not_a_foreign_key() -> None:
    # Deliberate design choice (see app/models/audit.py docstring): audit
    # rows must never be affected by a patients-table cascade.
    table = Base.metadata.tables["audit_logs"]

    assert not table.columns["patient_id"].foreign_keys


def test_citation_chunk_fk_uses_restrict_not_cascade() -> None:
    table = Base.metadata.tables["citations"]

    for fk in table.foreign_keys:
        if fk.parent.name == "chunk_id":
            assert fk.ondelete == "RESTRICT"
            return

    raise AssertionError("citations.chunk_id has no foreign key")
