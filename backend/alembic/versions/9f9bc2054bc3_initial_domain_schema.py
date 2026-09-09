"""initial domain schema

Creates the full persistent domain schema described in Milestone C:
users, patients, fhir_resources, clinical_documents, document_chunks,
document_embeddings, queries, retrieval_results, generated_responses,
citations, evaluation_results, audit_logs.

This is additive only - it creates new tables and does not touch, drop,
or alter any existing data. It assumes PostgreSQL with the ``vector``
extension available (already created by
``docker/postgres/init.sql`` on first container start); the
``CREATE EXTENSION IF NOT EXISTS vector`` statement here is a second,
idempotent safety net for environments where a fresh volume wasn't used
(it is a no-op if the extension already exists).

Revision ID: 9f9bc2054bc3
Revises:
Create Date: 2026-09-09
"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

from app.core.constants import DEFAULT_EMBEDDING_DIMENSIONS

# revision identifiers, used by Alembic.
revision: str = "9f9bc2054bc3"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Idempotent - safe even though docker/postgres/init.sql already does
    # this on a fresh volume. Never drops/recreates the extension.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column(
            "role",
            sa.Enum("ADMIN", "CLINICIAN", "REVIEWER", name="user_role"),
            nullable=False,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "patients",
        sa.Column("patient_id", sa.String(length=64), primary_key=True),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("sex", sa.String(length=32), nullable=True),
        sa.Column("birth_date", sa.Date(), nullable=True),
        sa.Column(
            "synthetic",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("patient_id", name="pk_patients"),
    )

    op.create_table(
        "fhir_resources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("patient_id", sa.String(length=64), nullable=False),
        sa.Column("resource_type", sa.String(length=64), nullable=False),
        sa.Column("resource_id", sa.String(length=128), nullable=False),
        sa.Column("raw_resource", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.patient_id"],
            name="fk_fhir_resources_patient_id_patients",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_fhir_resources"),
        sa.UniqueConstraint(
            "patient_id",
            "resource_type",
            "resource_id",
            name="uq_fhir_resource_identity",
        ),
    )
    op.create_index(
        "ix_fhir_resources_patient_id", "fhir_resources", ["patient_id"]
    )

    op.create_table(
        "clinical_documents",
        sa.Column("document_id", sa.String(length=128), primary_key=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("organization", sa.String(length=255), nullable=False),
        sa.Column("source_url", sa.String(length=2048), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=True),
        sa.Column("publication_date", sa.Date(), nullable=True),
        sa.Column("access_date", sa.Date(), nullable=False),
        sa.Column("license_or_usage_note", sa.Text(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("document_id", name="pk_clinical_documents"),
    )
    op.create_index(
        "ix_clinical_documents_category", "clinical_documents", ["category"]
    )

    op.create_table(
        "document_chunks",
        sa.Column("chunk_id", sa.String(length=256), primary_key=True),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("section_id", sa.String(length=64), nullable=True),
        sa.Column("section_heading", sa.String(length=512), nullable=True),
        sa.Column("chunk_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["clinical_documents.document_id"],
            name="fk_document_chunks_document_id_clinical_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id", name="pk_document_chunks"),
        sa.UniqueConstraint(
            "document_id", "chunk_number", name="uq_chunk_document_number"
        ),
    )
    op.create_index(
        "ix_document_chunks_document_id", "document_chunks", ["document_id"]
    )
    op.create_index(
        "ix_document_chunks_document_chunk_number",
        "document_chunks",
        ["document_id", "chunk_number"],
    )

    op.create_table(
        "document_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("chunk_id", sa.String(length=256), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(DEFAULT_EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.chunk_id"],
            name="fk_document_embeddings_chunk_id_document_chunks",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_embeddings"),
        sa.UniqueConstraint(
            "chunk_id", "model", name="uq_embedding_chunk_model"
        ),
    )
    op.create_index(
        "ix_document_embeddings_chunk_id", "document_embeddings", ["chunk_id"]
    )
    # IVFFlat cosine-distance index. Deliberately NOT created here: IVFFlat
    # index quality depends on having a representative sample of rows
    # already loaded (its lists are trained from existing data), so
    # building it against an empty table is not useful. Create it after
    # the first real ingestion+embedding run, e.g.:
    #   CREATE INDEX ix_document_embeddings_embedding_cosine
    #     ON document_embeddings
    #     USING ivfflat (embedding vector_cosine_ops)
    #     WITH (lists = 100);
    # Tracked in LOCAL_INTEGRATION_PENDING.md.

    op.create_table(
        "queries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("patient_id", sa.String(length=64), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("question_category", sa.String(length=64), nullable=True),
        sa.Column("evidence_status", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_queries_user_id_users",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.patient_id"],
            name="fk_queries_patient_id_patients",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_queries"),
    )
    op.create_index("ix_queries_request_id", "queries", ["request_id"])
    op.create_index("ix_queries_patient_id", "queries", ["patient_id"])

    op.create_table(
        "retrieval_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", sa.String(length=256), nullable=False),
        sa.Column("retrieval_method", sa.String(length=32), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("reranker_score", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["query_id"],
            ["queries.id"],
            name="fk_retrieval_results_query_id_queries",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.chunk_id"],
            name="fk_retrieval_results_chunk_id_document_chunks",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_retrieval_results"),
        sa.UniqueConstraint(
            "query_id",
            "chunk_id",
            "retrieval_method",
            name="uq_retrieval_result_identity",
        ),
    )
    op.create_index(
        "ix_retrieval_results_query_id", "retrieval_results", ["query_id"]
    )
    op.create_index(
        "ix_retrieval_results_chunk_id", "retrieval_results", ["chunk_id"]
    )

    op.create_table(
        "generated_responses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column(
            "key_points",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "warnings",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "safety_flags",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["query_id"],
            ["queries.id"],
            name="fk_generated_responses_query_id_queries",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_generated_responses"),
        sa.UniqueConstraint("query_id", name="uq_generated_responses_query_id"),
    )

    op.create_table(
        "citations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "generated_response_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column("chunk_id", sa.String(length=256), nullable=False),
        sa.Column("citation_label", sa.String(length=16), nullable=False),
        sa.Column("quoted_support", sa.Text(), nullable=False),
        sa.Column("support_classification", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["generated_response_id"],
            ["generated_responses.id"],
            name="fk_citations_generated_response_id_generated_responses",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.chunk_id"],
            name="fk_citations_chunk_id_document_chunks",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_citations"),
        sa.UniqueConstraint(
            "generated_response_id",
            "citation_label",
            name="uq_citation_label_per_response",
        ),
    )
    op.create_index(
        "ix_citations_generated_response_id",
        "citations",
        ["generated_response_id"],
    )

    op.create_table(
        "evaluation_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("benchmark_item_id", sa.String(length=128), nullable=True),
        sa.Column("query_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metric_name", sa.String(length=64), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("chunking_config", sa.String(length=64), nullable=True),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["query_id"],
            ["queries.id"],
            name="fk_evaluation_results_query_id_queries",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_results"),
    )
    op.create_index(
        "ix_evaluation_results_run_id", "evaluation_results", ["run_id"]
    )
    op.create_index(
        "ix_evaluation_results_run_metric",
        "evaluation_results",
        ["run_id", "metric_name"],
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("role", sa.String(length=32), nullable=True),
        sa.Column("patient_id", sa.String(length=64), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column(
            "retrieved_document_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("output_status", sa.String(length=32), nullable=True),
        sa.Column("evidence_sufficiency", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column(
            "safety_flags",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("hallucination_rate", sa.Float(), nullable=True),
        sa.Column("citation_correctness", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_audit_logs_user_id_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_audit_logs"),
    )
    op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"])
    op.create_index("ix_audit_logs_patient_id", "audit_logs", ["patient_id"])
    op.create_index(
        "ix_audit_logs_request_id_created",
        "audit_logs",
        ["request_id", "created_at"],
    )


def downgrade() -> None:
    # Reverse dependency order. Does not touch the `vector` extension -
    # other schemas/tables outside this migration's scope may depend on
    # it, so downgrading this migration must not drop it.
    op.drop_table("audit_logs")
    op.drop_table("evaluation_results")
    op.drop_table("citations")
    op.drop_table("generated_responses")
    op.drop_table("retrieval_results")
    op.drop_table("queries")
    op.drop_table("document_embeddings")
    op.drop_table("document_chunks")
    op.drop_table("clinical_documents")
    op.drop_table("fhir_resources")
    op.drop_table("patients")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS user_role")
