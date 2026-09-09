from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_TOP_K,
    DEFAULT_RRF_K,
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
