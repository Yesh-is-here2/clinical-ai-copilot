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


# --- Clinical LangGraph / grounded generation (Milestone E) ------------
#
# These mirror the constructor defaults already defined on the
# corresponding dataclasses (EvidenceSufficiencyConfig, ConfidenceConfig,
# ClinicalGraphDependencies) - having them here as well lets
# app.core.config.Settings expose them as env-configurable without
# duplicating the actual threshold *values* in two places by accident:
# both the dataclass defaults and the Settings defaults below read from
# this one module.

# Evidence sufficiency (app.clinical.evidence_sufficiency).
DEFAULT_EVIDENCE_SUFFICIENT_THRESHOLD = 0.65
DEFAULT_EVIDENCE_PARTIAL_THRESHOLD = 0.35
DEFAULT_EVIDENCE_TARGET_COUNT = 3
DEFAULT_EVIDENCE_TARGET_SOURCE_COUNT = 2

# Confidence scoring (app.clinical.confidence).
DEFAULT_CONFIDENCE_HIGH_THRESHOLD = 0.80
DEFAULT_CONFIDENCE_MODERATE_THRESHOLD = 0.60
DEFAULT_SEVERE_HALLUCINATION_RATE_THRESHOLD = 0.30

# Grounded generation (app.clinical.llm / app.clinical.graph). Not one of
# the embedding-model constants above - this is a chat/completions model
# used for structured clinical generation, a separate concern from
# DEFAULT_EMBEDDING_MODEL.
DEFAULT_GENERATION_MODEL = "gpt-4o-mini"
DEFAULT_GENERATION_TEMPERATURE = 0.0
DEFAULT_GENERATION_TIMEOUT_SECONDS = 30.0

# Regeneration is capped at exactly one retry by default (Phase 5
# constraint: "maximum one regeneration attempt globally for the
# request") - see app.clinical.graph._route_after_citation_validation,
# which makes a second, unbounded loop structurally impossible regardless
# of this value.
DEFAULT_MAX_REGENERATION_ATTEMPTS = 1
