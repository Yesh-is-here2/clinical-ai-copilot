# Authentication and RBAC — Phase 7

How authentication and authorization work in this service, and the
exact permission matrix. (Spec: master prompt, Phase 7.)

## Architecture

- **Tokens:** stateless JWTs, HS256, signed with `JWT_SECRET`
  (see `.env.example`). Default lifetime 30 minutes
  (`ACCESS_TOKEN_EXPIRE_MINUTES`). No refresh tokens and no logout
  endpoint: the chosen token architecture is stateless, so neither is
  required (spec 7.1: add them only if the architecture genuinely
  requires them).
- **Passwords:** bcrypt (`hash_password` / `verify_password` in
  `app/core/security.py`). No passlib (unmaintained). Passwords are
  never logged, never stored in plaintext, and never appear in any
  API response. Bcrypt's 72-byte limit is enforced at creation
  (rejected, never truncated).
- **Identity:** `get_current_actor` (`app/api/deps.py`) validates the
  Bearer token, loads the live `users` row, and rejects missing,
  expired, malformed, wrongly-signed tokens, unknown users, and
  deactivated accounts with 401. The actor's role always comes from
  the database row, never from the token's role claim, so a role
  change or deactivation takes effect on the next request.
- **Authorization:** `require_roles(...)` gates roles (403). Every
  protected route enforces its rule in the backend; nothing relies
  on the frontend hiding anything.
- **Users:** the `users` table already existed (initial migration);
  Phase 7 added no migration. Admins create users via
  `POST /auth/users`; there is no public registration by design.
- **Login hardening:** failed attempts are rate-limited per
  (client IP, email) — 10/minute by default
  (`LOGIN_RATE_LIMIT_MAX_ATTEMPTS` /
  `LOGIN_RATE_LIMIT_WINDOW_SECONDS`); a successful login clears the
  counter. A wrong email, wrong password, and deactivated account
  all produce the same 401, and a dummy bcrypt check runs for
  unknown emails so timing does not leak whether an account exists.
- **Production secrets:** startup refuses to serve when
  `APP_ENV=production` and the JWT secret is missing, a known
  placeholder, or shorter than 32 characters, or the algorithm is
  not HS256 (`validate_security_settings`, called from the lifespan).
  Development/testing tolerate the insecure default so the app and
  the test suite work with no `.env` at all.
- **Transport controls:** CORS is closed by default (only installed
  when `CORS_ALLOW_ORIGINS` is configured, and then only for the
  named origins); security headers on every response
  (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`), plus `Cache-Control: no-store`
  on `/auth/*`; request bodies are capped at 1 MB (413 above).
- **Audit:** every persisted clinical query and audit row carries
  the actor (`user_id`, `role`). No password, token, or secret is
  ever written to an audit row.

## Permission matrix

| Endpoint | Anonymous | Clinician | Reviewer | Admin |
|---|:-:|:-:|:-:|:-:|
| `GET /` | ✅ | ✅ | ✅ | ✅ |
| `GET /health` | ✅ | ✅ | ✅ | ✅ |
| `POST /auth/login` | ✅ | ✅ | ✅ | ✅ |
| `GET /auth/me` | ❌ 401 | ✅ | ✅ | ✅ |
| `GET /auth/users` | ❌ 401 | ❌ 403 | ❌ 403 | ✅ |
| `POST /auth/users` | ❌ 401 | ❌ 403 | ❌ 403 | ✅ |
| `GET /patients` (+ `/{id}`) | ❌ 401 | ✅ | ✅ | ✅ |
| `POST /clinical/query` | ❌ 401 | ✅ | ❌ 403 | ✅ |
| `GET /documents` | ❌ 401 | ✅ | ✅ | ✅ |
| `POST /documents/ingest` | ❌ 401 | ❌ 403 | ❌ 403 | ✅ |
| `GET /queries/{id}` | ❌ 401 | own only¹ | own only¹ | ✅ all |
| `GET /queries/{id}/evaluation` | ❌ 401 | ❌ 403 | ✅ | ✅ |
| `GET /audit` | ❌ 401 | ❌ 403 | ✅ | ✅ |

¹ A clinician/reviewer sees only queries they submitted, plus legacy
rows with no owner (which predate authentication). Another user's
query is indistinguishable from a missing one (404 either way), so
query ids cannot be probed.

Notes:

- `GET /` and `GET /health` are public by design: the root returns
  only service links, and the health check returns only
  healthy/connected/available status — no clinical or user data.
- Reviewers cannot submit clinical queries: spec 7.5 allows it only
  "if policy explicitly allows", and no such policy exists, so least
  privilege denies it.
- `POST /evaluation/run` and `GET /metrics/summary` are Phase 8/10
  scope and do not exist yet; when they land they are
  reviewer/admin-only per spec 7.5.
- Denied requests (401/403) are visible in the request log
  (method/path/status/request id) but are not written to the
  `audit_logs` table: that table records completed clinical
  operations, with the actor attached.

## Verification

- Unit suite (no database, no network):
  `DATABASE_URL=postgresql+asyncpg://u:p@localhost:5432/db pytest -q`
  (from `backend/`, with proxy env vars unset — see the repo notes on
  the sandbox proxy breaking the OpenAI client constructor).
- Integration suite (needs the compose database):
  `docker compose exec backend pytest -m integration -q`
