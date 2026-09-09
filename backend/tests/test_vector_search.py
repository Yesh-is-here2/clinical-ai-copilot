"""Unit tests for the semantic-search statement builder.

These compile the SQLAlchemy statement to text and assert on its shape -
no database connection involved. Actually executing a search against real
pgvector data is covered by
tests/integration/test_postgres_integration.py (requires PostgreSQL).
"""

from sqlalchemy.dialects import postgresql

from app.retrieval.vector_search import (
    VectorSearchError,
    VectorSearchService,
    build_vector_search_statement,
)


def compiled_sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_statement_uses_cosine_distance_operator() -> None:
    stmt = build_vector_search_statement(
        [0.1, 0.2, 0.3], model="text-embedding-3-small", top_k=5
    )

    sql = compiled_sql(stmt)

    assert "<=>" in sql


def test_statement_filters_by_embedding_model() -> None:
    stmt = build_vector_search_statement(
        [0.1, 0.2], model="text-embedding-3-small", top_k=5
    )

    sql = compiled_sql(stmt)

    assert "document_embeddings.model =" in sql


def test_statement_orders_by_distance_then_chunk_id_for_stable_ties() -> None:
    stmt = build_vector_search_statement([0.1], model="m", top_k=5)

    sql = compiled_sql(stmt)

    assert "ORDER BY distance ASC, document_embeddings.chunk_id ASC" in sql


def test_statement_applies_limit() -> None:
    stmt = build_vector_search_statement([0.1], model="m", top_k=7)

    sql = compiled_sql(stmt)

    assert "LIMIT" in sql


def test_statement_joins_chunk_and_document_tables() -> None:
    stmt = build_vector_search_statement([0.1], model="m", top_k=5)

    sql = compiled_sql(stmt)

    assert "JOIN document_chunks" in sql
    assert "JOIN clinical_documents" in sql


def test_rejects_zero_or_negative_top_k() -> None:
    for bad_top_k in (0, -1):
        try:
            build_vector_search_statement([0.1], model="m", top_k=bad_top_k)
        except VectorSearchError:
            continue
        raise AssertionError(f"top_k={bad_top_k} should have been rejected")


def test_rejects_empty_query_embedding() -> None:
    try:
        build_vector_search_statement([], model="m", top_k=5)
    except VectorSearchError:
        return
    raise AssertionError("empty embedding should have been rejected")


def test_service_rejects_non_positive_default_top_k() -> None:
    try:
        VectorSearchService(session=None, model="m", default_top_k=0)
    except VectorSearchError:
        return
    raise AssertionError("default_top_k=0 should have been rejected")
