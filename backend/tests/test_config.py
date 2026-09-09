import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.constants import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODEL_DIMENSIONS,
)


DATABASE_URL = "postgresql+asyncpg://u:p@localhost/db"


# ---------------------------------------------------------------------
# Blank env values must normalize to the Python field default, not win
# as an empty string. This is what `env_ignore_empty=True` (in
# Settings.model_config) fixes - a real .env / docker-compose.yml
# development environment ships optional settings as documented but
# blank placeholders (`OPENAI_API_KEY=`, `EMBEDDING_MODEL=`), and those
# must not shadow the code-level default.
# ---------------------------------------------------------------------


def test_blank_env_openai_api_key_normalizes_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = Settings(_env_file=None, database_url=DATABASE_URL)

    assert settings.openai_api_key is None


def test_blank_env_embedding_model_normalizes_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "")

    settings = Settings(_env_file=None, database_url=DATABASE_URL)

    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_nonblank_env_values_still_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # env_ignore_empty must only special-case the empty string - a real,
    # non-empty env value has to keep winning over the Python default,
    # exactly as before.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")

    settings = Settings(_env_file=None, database_url=DATABASE_URL)

    assert settings.openai_api_key == "sk-test-not-a-real-key"
    assert settings.embedding_model == "text-embedding-3-large"


def test_loading_the_real_dotenv_file_normalizes_blank_values() -> None:
    # This is the exact real-world scenario the bug came from: the
    # backend's actual .env (loaded via the default env_file=".env",
    # no override here) ships OPENAI_API_KEY= and EMBEDDING_MODEL=
    # blank. Constructing Settings() the normal way it must still
    # resolve to the Python defaults, not empty strings.
    settings = Settings(database_url=DATABASE_URL)

    assert settings.openai_api_key is None
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_required_database_url_is_still_required_when_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # env_ignore_empty must not weaken validation for fields that have
    # no default - a blank DATABASE_URL should still fail loudly rather
    # than silently resolve to something.
    monkeypatch.setenv("DATABASE_URL", "")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


# ---------------------------------------------------------------------
# Plain defaults (no env involved)
# ---------------------------------------------------------------------


def test_embedding_model_defaults_to_text_embedding_3_small() -> None:
    settings = Settings(_env_file=None, database_url=DATABASE_URL)

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_openai_api_key_defaults_to_none() -> None:
    settings = Settings(_env_file=None, database_url=DATABASE_URL)

    assert settings.openai_api_key is None


def test_semantic_and_bm25_top_k_have_sane_defaults() -> None:
    settings = Settings(_env_file=None, database_url=DATABASE_URL)

    assert settings.semantic_top_k == 15
    assert settings.bm25_top_k == 15


def test_top_k_settings_reject_non_positive_values() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            database_url=DATABASE_URL,
            semantic_top_k=0,
        )


def test_default_embedding_dimensions_matches_the_default_model() -> None:
    assert (
        DEFAULT_EMBEDDING_DIMENSIONS
        == EMBEDDING_MODEL_DIMENSIONS[DEFAULT_EMBEDDING_MODEL]
    )
    assert DEFAULT_EMBEDDING_DIMENSIONS == 1536
