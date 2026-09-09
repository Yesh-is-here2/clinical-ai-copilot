# Local integration pending

Tracks work that has been implemented and structurally validated (offline
SQL generation, mocked provider tests, in-memory fake repositories) but
has **not** been verified against a real PostgreSQL + pgvector database,
because the environment this code was written in has no route to your
Docker daemon. Nothing below should be treated as "done" until you've run
it and it's checked off.

## Milestone C - PostgreSQL / pgvector verification: RESOLVED / VERIFIED

Confirmed against a real local Docker + PostgreSQL + pgvector environment
(not the sandbox this was developed in):

- [x] **Migration applied and current.** `docker compose exec backend
      alembic upgrade head` / `alembic current` ran clean: all 12 domain
      tables created, `alembic_version` at head (`9f9bc2054bc3`), no
      errors, existing data untouched.

- [x] **Integration test suite runs against the real database.**
      `docker compose exec backend pytest -m integration -q` genuinely
      executes both tests against real PostgreSQL + pgvector (no
      skip/deselect, no mocked DB):
      - the embed -> persist -> pgvector cosine-distance search round
        trip
      - the embedding-persistence service's duplicate/re-ingestion skip
        logic against the real `SqlAlchemyEmbeddingRepository` (ON
        CONFLICT upsert)

      Verified passing on two separate runs (2 passed / 2 passed), with
      the `itest-*`-scoped cleanup in the `db_session` fixture confirmed
      repeatable and confirmed to leave the schema itself untouched
      (no drop/recreate, no Alembic downgrade) between runs.

      Two real bugs turned up during this verification and are now
      fixed (not just structurally validated):
      - the integration tests originally embedded with
        `FakeDeterministicEmbeddingProvider(dimensions=8)`, but
        `document_embeddings.embedding` is a real `vector(1536)` column
        - fixed to use `DEFAULT_EMBEDDING_DIMENSIONS`, and
        `SqlAlchemyEmbeddingRepository.upsert_many` now validates
        embedding dimensionality itself (`EmbeddingDimensionMismatchError`)
        before ever reaching asyncpg, with dedicated unit-test coverage
        in `tests/test_embedding_repository.py`.
      - the `db_session` fixture originally reused the application's
        module-global pooled engine across AnyIO's per-test event loops,
        causing "Future attached to a different loop" / "Event loop is
        closed" on the second test - fixed by giving the fixture its own
        `NullPool`-backed `AsyncEngine`, scoped and disposed within that
        one test's lifetime.

- [x] **`docker compose exec backend pytest -q` full non-integration
      count confirmed** inside the real container: 181 passed, 2
      deselected, 2 known non-blocking warnings.

- [x] **`git diff --check` reviewed.** No actual whitespace errors; only
      Windows LF/CRLF warnings (expected on this checkout, not a code
      issue).

## Milestone C - still open (not part of the verification above)

- [ ] **Build the IVFFlat index once real data exists.** The migration
      intentionally does not create a vector index (an IVFFlat index
      trained against zero rows is not useful). After a real
      ingestion+embedding run has populated `document_embeddings`, run:

      ```sql
      CREATE INDEX ix_document_embeddings_embedding_cosine
        ON document_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
      ```

      Adjust `lists` per pgvector's guidance (roughly `rows / 1000` for
      larger corpora); 100 is a reasonable starting point for the small
      guideline corpus this project indexes.

- [ ] **Real OpenAI embedding call**, if/when you have a configured
      `OPENAI_API_KEY`: `OpenAIEmbeddingProvider` has only been tested
      against a mocked client (`tests/test_embeddings.py`) and against
      the deterministic fake provider in the integration suite above. A
      single real call (e.g. embedding one ingested chunk) would confirm
      the live `text-embedding-3-small` response shape still matches
      what the mocks assume (1536-dim vectors,
      `response.data[i].embedding`).

Once the two items above are checked off (or deliberately deferred past
Milestone C), this file's Milestone C sections can be deleted, or a
Milestone D section added in their place.
