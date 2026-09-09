"""Deterministic prompt-injection detection (Milestone E, section 5).

Scans free text (the user's question) for phrases/patterns that attempt
to override system instructions, exfiltrate the system prompt, or
otherwise hijack the clinical assistant's behavior. Detection is
pattern-based and deterministic - no LLM call, no network - so it runs
before any retrieval or generation happens and never depends on an
external provider being available.

This module only detects; it does not itself decide to block a request
- see app.clinical.graph, which routes to a safe "blocked" response when
``InjectionDetectionResult.detected`` is True. Retrieved evidence text is
handled separately (see app.clinical.prompting): it is always delimited
as untrusted data in the generation prompt, never re-scanned by this
detector as if it might contain user instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class InjectionSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_SEVERITY_ORDER = {
    InjectionSeverity.LOW: 0,
    InjectionSeverity.MEDIUM: 1,
    InjectionSeverity.HIGH: 2,
}


@dataclass(frozen=True)
class InjectionRule:
    name: str
    pattern: re.Pattern[str]
    severity: InjectionSeverity


def _rule(name: str, pattern: str, severity: InjectionSeverity) -> InjectionRule:
    return InjectionRule(
        name=name, pattern=re.compile(pattern, re.IGNORECASE), severity=severity
    )


# Patterns are deliberately anchored to whole phrases (not bare keywords
# like "ignore" or "pretend") so ordinary clinical language - "ignore my
# previous blood pressure reading", "pretend I didn't take my dose
# yesterday" - does not false-positive. See tests/test_injection.py for
# the false-positive cases this is checked against.
_RULES: tuple[InjectionRule, ...] = (
    _rule(
        "ignore_previous_instructions",
        r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions?",
        InjectionSeverity.HIGH,
    ),
    _rule(
        "ignore_clinical_guidelines",
        r"ignore\s+(the\s+|all\s+)?clinical\s+guidelines?",
        InjectionSeverity.HIGH,
    ),
    _rule(
        "reveal_system_prompt",
        r"reveal\s+(the\s+|your\s+)?system\s+prompt",
        InjectionSeverity.HIGH,
    ),
    _rule(
        "print_system_prompt",
        r"print\s+(the\s+|your\s+|out\s+)?system\s+prompt",
        InjectionSeverity.HIGH,
    ),
    _rule(
        "pretend_you_are",
        r"pretend\s+(that\s+)?you\s+are\s+(a|an)\b",
        InjectionSeverity.MEDIUM,
    ),
    _rule(
        "bypass_safety",
        r"bypass\s+(the\s+|all\s+)?safety",
        InjectionSeverity.HIGH,
    ),
    _rule(
        "disregard_the_evidence",
        r"disregard\s+(the\s+|all\s+)?evidence",
        InjectionSeverity.MEDIUM,
    ),
    _rule(
        "follow_instructions_in_documents",
        r"follow\s+(the\s+)?instructions?\s+(in|within|inside)\s+"
        r"(the\s+)?(retrieved\s+)?documents?",
        InjectionSeverity.MEDIUM,
    ),
)


_SAFE_MESSAGE = (
    "This request could not be processed because it appears to contain "
    "an attempt to override this system's instructions or safety "
    "behavior. Please rephrase your clinical question without asking "
    "the assistant to ignore its instructions, reveal internal "
    "configuration, or bypass safety checks."
)


@dataclass(frozen=True)
class InjectionDetectionResult:
    detected: bool
    matched_rules: list[str] = field(default_factory=list)
    severity: InjectionSeverity | None = None
    safe_message: str | None = None


def detect_prompt_injection(text: str) -> InjectionDetectionResult:
    """Deterministically scan ``text`` for known injection patterns.

    Returns a structured result rather than a bare bool so callers (the
    graph, audit logging) get the matched rule names and a severity, not
    just yes/no - useful for both routing and for explaining a block to
    the user without echoing their raw payload back at them.
    """

    if not text:
        return InjectionDetectionResult(detected=False)

    matched = [rule.name for rule in _RULES if rule.pattern.search(text)]

    if not matched:
        return InjectionDetectionResult(detected=False)

    matched_severities = [
        rule.severity for rule in _RULES if rule.name in matched
    ]
    highest_severity = max(matched_severities, key=lambda s: _SEVERITY_ORDER[s])

    return InjectionDetectionResult(
        detected=True,
        matched_rules=matched,
        severity=highest_severity,
        safe_message=_SAFE_MESSAGE,
    )
