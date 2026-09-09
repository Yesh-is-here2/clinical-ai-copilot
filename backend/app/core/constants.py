"""Small, stable constants shared across modules.

These are deliberately plain module constants, not Pydantic ``Settings``
fields: ``DEFAULT_EMBEDDING_DIMENSIONS`` is baked into the
``document_embeddings.embedding`` pgvector column at migration time (see
``alembic/versions``), so it cannot be a value that silently changes
because someone edited ``.env`` - changing the embedding model's
dimensionality is a schema change and must go through a new migration,
not an environment variable.
"""

# Known OpenAI embedding model output dimensionality. Used to validate
# that an embedding provider's output actually matches what the
# `document_embeddings` table was migrated to store, instead of letting a
# dimension mismatch surface later as an opaque database error.
EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# The dimensionality actually baked into the pgvector column. Kept as its
# own named constant (rather than a dict lookup scattered through model
# code) so it is one clear import: `from app.core.constants import
# DEFAULT_EMBEDDING_DIMENSIONS`.
DEFAULT_EMBEDDING_DIMENSIONS = EMBEDDING_MODEL_DIMENSIONS[DEFAULT_EMBEDDING_MODEL]
