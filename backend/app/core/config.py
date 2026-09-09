from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    DEFAULT_CONFIDENCE_HIGH_THRESHOLD,
    DEFAULT_CONFIDENCE_MODERATE_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EVIDENCE_PARTIAL_THRESHOLD,
    DEFAULT_EVIDENCE_SUFFICIENT_THRESHOLD,
    DEFAULT_EVIDENCE_TARGET_COUNT,
    DEFAULT_EVIDENCE_TARGET_SOURCE_COUNT,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_GENERATION_TEMPERATURE,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_MAX_REGENERATION_ATTEMPTS,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_TOP_K,
    DEFAULT_RRF_K,
    DEFAULT_SEVERE_HALLUCINATION_RATE_THRESHOLD,
)


class Settings(BaseSettings):
    app_name: str = Field(default="Clinical AI Copilot")
    app_env: Literal["development", "testing", "staging", "production"] = "development"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    database_url: str

    # --- Embeddings / retrieval (Milestone C) -----------------------
    # No default for the API key: it is genuinely optional (many routes,
    # and the whole test suite, must work without it) but code that
    # actually needs to call OpenAI must fail explicitly rather than
    # silently proceed - see app.services.embeddings.
    openai_api_key: str | None = Field(default=None)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    semantic_top_k: int = Field(default=15, gt=0)
    bm25_top_k: int = Field(default=15, gt=0)

    # --- Hybrid retrieval / reranking (Milestone D) -------------------
    rrf_k: int = Field(default=DEFAULT_RRF_K, gt=0)
    reranker_top_k: int = Field(default=DEFAULT_RERANKER_TOP_K, gt=0)
    reranker_model: str = Field(default=DEFAULT_RERANKER_MODEL)
    # Whether hybrid retrieval should attempt reranking at all. Note this
    # is independent of whether a real reranker can actually be loaded -
    # the optional sentence-transformers/torch dependency (see
    # backend/requirements-reranker.txt) may not be installed even when
    # this is True, in which case HybridRetrievalService falls back to
    # fused-only results with explicit degradation metadata rather than
    # failing the request. Set this False to skip reranking by design
    # (not as a failure) instead.
    reranker_enabled: bool = Field(default=True)

    # --- Clinical LangGraph / grounded generation (Milestone E) -------
    evidence_sufficient_threshold: float = Field(
        default=DEFAULT_EVIDENCE_SUFFICIENT_THRESHOLD, ge=0.0, le=1.0
    )
    evidence_partial_threshold: float = Field(
        default=DEFAULT_EVIDENCE_PARTIAL_THRESHOLD, ge=0.0, le=1.0
    )
    evidence_target_count: int = Field(default=DEFAULT_EVIDENCE_TARGET_COUNT, gt=0)
    evidence_target_source_count: int = Field(
        default=DEFAULT_EVIDENCE_TARGET_SOURCE_COUNT, gt=0
    )

    confidence_high_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_HIGH_THRESHOLD, ge=0.0, le=1.0
    )
    confidence_moderate_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_MODERATE_THRESHOLD, ge=0.0, le=1.0
    )
    severe_hallucination_rate_threshold: float = Field(
        default=DEFAULT_SEVERE_HALLUCINATION_RATE_THRESHOLD, ge=0.0, le=1.0
    )

    # Chat/completions model used for structured clinical generation -
    # deliberately separate from `embedding_model` above (a different
    # OpenAI API surface). Uses the same `openai_api_key` field; no
    # additional secret is introduced for Milestone E.
    generation_model: str = Field(default=DEFAULT_GENERATION_MODEL)
    generation_temperature: float = Field(default=DEFAULT_GENERATION_TEMPERATURE, ge=0.0)
    generation_timeout_seconds: float = Field(
        default=DEFAULT_GENERATION_TIMEOUT_SECONDS, gt=0.0
    )
    max_regeneration_attempts: int = Field(default=DEFAULT_MAX_REGENERATION_ATTEMPTS, ge=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # `.env.example`/`.env` intentionally ship optional settings as
        # blank placeholders (OPENAI_API_KEY=, EMBEDDING_MODEL=) so the
        # file documents every recognized variable even when unset. This
        # tells pydantic-settings to treat "" as "not provided" for those
        # variables, so the Python field default (None /
        # DEFAULT_EMBEDDING_MODEL) applies instead of an empty string
        # silently winning. A genuinely non-empty env value still
        # overrides the default normally - this only changes how a
        # blank value is treated.
        env_ignore_empty=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
