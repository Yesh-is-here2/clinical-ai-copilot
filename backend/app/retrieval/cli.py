"""Controlled command-line entry point for guideline ingestion.

This lives inside ``app/`` (not a root-level ``scripts/`` directory) on
purpose: the backend Docker image's build context is ``./backend``, so a
script living outside it would not exist inside the container. Run it as
a module so relative imports resolve the same way they do for the API:

    python -m app.retrieval.cli ingest <document-id>
    python -m app.retrieval.cli ingest who-hypertension-pharmacological-2021 \\
        --output-dir /app/data/guidelines/processed

It deliberately does not reimplement any ingestion logic - it is a thin
wrapper around ``get_guideline_source``, ``ingest_guideline`` and
``write_ingestion_manifest``, which are the same functions the future
``POST /documents/ingest`` API route will call. Exit codes are distinct
per failure class so calling scripts/CI can distinguish "bad input" from
"network/extraction failure" from "couldn't write output":

    0  success
    2  unknown/unregistered document id
    3  download or extraction failure
    4  manifest could not be persisted
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.retrieval.ingestion import GuidelineIngestionError, ingest_guideline
from app.retrieval.manifest import ManifestPersistenceError, write_ingestion_manifest
from app.retrieval.sources import get_guideline_source

DEFAULT_OUTPUT_DIR = Path("/app/data/guidelines/processed")

EXIT_OK = 0
EXIT_UNKNOWN_SOURCE = 2
EXIT_INGESTION_FAILED = 3
EXIT_MANIFEST_FAILED = 4


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval.cli",
        description=(
            "Ingest a single registered clinical guideline source: "
            "download it, extract normalized sections, and persist a "
            "deterministic ingestion manifest."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest one registered guideline source by document ID.",
    )
    ingest_parser.add_argument(
        "document_id",
        help=(
            "Registered guideline document ID, e.g. "
            "'who-hypertension-pharmacological-2021'."
        ),
    )
    ingest_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory to write the ingestion manifest into "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )

    return parser


async def _run_ingest(document_id: str, output_dir: Path) -> int:
    source = get_guideline_source(document_id)

    if source is None:
        print(
            f"error: unknown guideline source '{document_id}'. "
            "It is not registered in app.retrieval.sources.",
            file=sys.stderr,
        )
        return EXIT_UNKNOWN_SOURCE

    try:
        result = await ingest_guideline(source)
    except GuidelineIngestionError as exc:
        print(f"error: ingestion failed for '{document_id}': {exc}", file=sys.stderr)
        return EXIT_INGESTION_FAILED

    try:
        manifest_path = write_ingestion_manifest(result, output_dir)
    except ManifestPersistenceError as exc:
        print(
            f"error: failed to persist manifest for '{document_id}': {exc}",
            file=sys.stderr,
        )
        return EXIT_MANIFEST_FAILED

    summary = {
        "document_id": result.document.metadata.document_id,
        "title": result.document.metadata.title,
        "organization": result.document.metadata.organization,
        "status": result.status.value,
        "section_count": result.provenance.section_count,
        "content_length": result.provenance.content_length,
        "sha256": result.provenance.sha256,
        "manifest_path": str(manifest_path),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return asyncio.run(
            _run_ingest(args.document_id, args.output_dir)
        )

    parser.error(f"Unknown command '{args.command}'.")
    return EXIT_UNKNOWN_SOURCE  # pragma: no cover - argparse exits first


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
