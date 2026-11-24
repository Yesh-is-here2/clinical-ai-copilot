"""Tiny in-memory sliding-window limiter for login attempts
(Phase 7, section 7.6).

Counts *failed* attempts per key (``"<client ip>|<email>"``) in a
rolling window. A successful login clears the key, so a legitimate
user who mistypes once is never locked out by their own earlier
failures, and a distributed attack still has to guess within the
window from each (ip, email) pair.

This is deliberately process-local and best-effort: it exists to
slow brute force on a single-instance deployment, not to be a
distributed rate-limiting system. State is intentionally never
persisted and never logged.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class LoginRateLimiter:
    """Sliding-window failed-login limiter."""

    def __init__(self, *, max_attempts: int, window_seconds: int) -> None:
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        """Drop failures older than the window; drop the key entirely
        when nothing is left."""

        attempts = self._failures.get(key)
        if attempts is None:
            attempts = deque()
            self._failures[key] = attempts
        cutoff = now - self._window_seconds
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        return attempts

    def is_allowed(self, key: str) -> bool:
        """True while the key still has attempts left in the window."""

        with self._lock:
            return len(self._prune(key, time.monotonic())) < self._max_attempts

    def record_failure(self, key: str) -> None:
        """Count one failed attempt for the key."""

        with self._lock:
            self._prune(key, time.monotonic()).append(time.monotonic())

    def record_success(self, key: str) -> None:
        """Clear the key: a good login resets the counter."""

        with self._lock:
            self._failures.pop(key, None)
