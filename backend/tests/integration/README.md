# Integration tests

Everything in this directory is marked `@pytest.mark.integration` and is
**excluded from the default test run** (see `backend/pytest.ini`,
`addopts = -m "not integration"`). These tests require a live PostgreSQL
instance with the `vector` extension - the same database
`docker-compose.yml` brings up - and are not runnable in a sandboxed
environment with no route to that database.

Run them explicitly, against a real database, with:

```powershell
docker compose exec backend pytest -m integration -q
```

Do not remove the `integration` marker to make these "pass" in a
sandbox - that would mean either mocking pgvector/Postgres behavior
(explicitly against the project's rules) or silently not testing the
real thing at all.

See `LOCAL_INTEGRATION_PENDING.md` at the repo root for the current list
of Milestone C items that specifically need a real-database run before
they're verified.
