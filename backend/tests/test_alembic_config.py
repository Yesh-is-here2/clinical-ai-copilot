"""Structural validation of the Alembic setup - no live database.

Runs Alembic's *offline* SQL-generation mode, which only needs a dialect
(derived from the configured DATABASE_URL) to compile DDL text; it never
opens a socket. This is exactly the "structurally validate Alembic
configuration" / "test migration metadata/offline generation" requirement
from Milestone C - actually executing `alembic upgrade head` against a
live database is tracked in LOCAL_INTEGRATION_PENDING.md instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _run_alembic(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_alembic_ini_and_env_py_exist() -> None:
    assert (BACKEND_DIR / "alembic.ini").exists()
    assert (BACKEND_DIR / "alembic" / "env.py").exists()


def test_exactly_one_initial_migration_exists() -> None:
    versions_dir = BACKEND_DIR / "alembic" / "versions"
    migration_files = sorted(versions_dir.glob("*.py"))

    assert len(migration_files) == 1
    assert "initial_domain_schema" in migration_files[0].name


def test_alembic_history_runs_offline() -> None:
    # `history` never needs a live connection - a bare structural sanity
    # check that env.py at least imports and Alembic can read the
    # migration chain.
    result = _run_alembic("history")

    assert result.returncode == 0, result.stderr
    assert "initial domain schema" in result.stdout


def test_alembic_offline_upgrade_sql_creates_all_tables() -> None:
    result = _run_alembic("upgrade", "head", "--sql")

    assert result.returncode == 0, result.stderr

    expected_tables = [
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
    ]

    for table_name in expected_tables:
        assert f"CREATE TABLE {table_name} " in result.stdout, table_name

    assert "CREATE EXTENSION IF NOT EXISTS vector" in result.stdout
    assert "VECTOR(1536)" in result.stdout


def test_alembic_offline_upgrade_sql_never_drops_anything() -> None:
    result = _run_alembic("upgrade", "head", "--sql")

    assert result.returncode == 0, result.stderr
    assert "DROP TABLE" not in result.stdout
    assert "DROP EXTENSION" not in result.stdout


def test_alembic_offline_downgrade_sql_is_well_formed() -> None:
    result = _run_alembic(
        "downgrade", "9f9bc2054bc3:base", "--sql"
    )

    assert result.returncode == 0, result.stderr
    assert "DROP TABLE users" in result.stdout
    # Downgrading must never drop the pgvector extension - other
    # schemas/tables outside this migration's scope may depend on it.
    assert "DROP EXTENSION" not in result.stdout
