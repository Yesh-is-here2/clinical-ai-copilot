"""Deterministic clinical safety layer (Milestone E, section 13).

Pattern-based, not LLM-based - runs in every default test with no
provider needed. Patterns are anchored to specific directive phrasing
("you have...", "take X mg", "you don't need to see a doctor") rather
than bare keywords, so describing what a guideline recommends in general
terms ("guidelines recommend a target below 130/80 mmHg") does not
trip these checks - see tests/test_safety.py's neutral-educational-
content cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class SafetySeverity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_SEVERITY_ORDER = {
    SafetySeverity.NONE: 0,
    SafetySeverity.LOW: 1,
    SafetySeverity.MEDIUM: 2,
    SafetySeverity.HIGH: 3,
}


@dataclass(frozen=True)
class SafetyFlag:
    rule: str
    severity: SafetySeverity
    matched_text: str | None = None


@dataclass(frozen=True)
class SafetyCheckResult:
    passed: bool
    severity: SafetySeverity
    flags: list[SafetyFlag] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    guarded_response_required: bool = False


def _rule(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_SAFETY_RULES: tuple[tuple[str, re.Pattern[str], SafetySeverity], ...] = (
    (
        "unsupported_diagnosis_assertion",
        _rule(r"\byou (have|are diagnosed with|are suffering from)\b"),
        SafetySeverity.HIGH,
    ),
    (
        "definitive_treatment_instruction",
        _rule(
            r"\byou (should|must) take\b|"
            r"\btake \d+(\.\d+)?\s*(mg|mcg|milligrams|micrograms|ml)\b|"
            r"\b(start|stop) taking\b"
        ),
        SafetySeverity.HIGH,
    ),
    (
        "unsupported_dosage_change",
        _rule(
            r"\b(increase|decrease|double|halve)\s+(your\s+)?(the\s+)?dos(e|age)\b"
        ),
        SafetySeverity.HIGH,
    ),
    (
        "unsafe_removal_of_clinician_involvement",
        _rule(
            r"\byou (don't|do not) need to (see|consult)\s+(a\s+)?"
            r"(doctor|physician|clinician|healthcare provider)\b|"
            r"\bno need to (see|consult)\s+(a\s+)?"
            r"(doctor|physician|clinician|healthcare provider)\b"
        ),
        SafetySeverity.HIGH,
    ),
    (
        "unsupported_emergency_reassurance",
        _rule(
            r"\bthis is not an emergency\b|"
            r"\byou (don't|do not) need to go to (the\s+)?"
            r"(er|emergency room|the emergency department)\b"
        ),
        SafetySeverity.HIGH,
    ),
)


def run_safety_checks(answer_text: str, *, disclaimer: str) -> SafetyCheckResult:
    flags: list[SafetyFlag] = []

    for rule_name, pattern, severity in _SAFETY_RULES:
        match = pattern.search(answer_text or "")

        if match:
            flags.append(
                SafetyFlag(rule=rule_name, severity=severity, matched_text=match.group(0))
            )

    if not disclaimer or not disclaimer.strip():
        flags.append(SafetyFlag(rule="missing_disclaimer", severity=SafetySeverity.LOW))

    if not flags:
        return SafetyCheckResult(
            passed=True, severity=SafetySeverity.NONE, guarded_response_required=False
        )

    highest_severity = max(flags, key=lambda flag: _SEVERITY_ORDER[flag.severity]).severity
    warnings = [f"Safety check flagged '{flag.rule}'." for flag in flags]

    return SafetyCheckResult(
        passed=False,
        severity=highest_severity,
        flags=flags,
        warnings=warnings,
        guarded_response_required=highest_severity
        in (SafetySeverity.MEDIUM, SafetySeverity.HIGH),
    )
