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


# --- Hybrid retrieval / reranking (Milestone D) -----------------------

# Standard Reciprocal Rank Fusion smoothing constant. The original RRF
# paper, and most production hybrid-search implementations, use k=60 -
# large enough that a result ranked #1 by only one retriever doesn't
# completely dominate the fused ranking over a result ranked well by
# both retrievers.
DEFAULT_RRF_K = 60

# How many reranked results a hybrid retrieval call returns by default.
DEFAULT_RERANKER_TOP_K = 5

# cross-encoder/ms-marco-MiniLM-L-6-v2: a small (~80MB), well-established
# MS MARCO-trained cross-encoder - a reasonable default for reranking
# short clinical-guideline passages without requiring a GPU. Not
# downloaded or imported anywhere at startup - see
# app.retrieval.reranker.CrossEncoderReranker, which loads it lazily and
# only when reranking is actually attempted.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
