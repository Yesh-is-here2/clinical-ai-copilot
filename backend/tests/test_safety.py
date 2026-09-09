"""Unit tests for app.clinical.safety (Milestone E, section 13)."""

from __future__ import annotations

import pytest

from app.clinical.safety import SafetySeverity, run_safety_checks


@pytest.mark.parametrize(
    "answer_text,expected_rule",
    [
        ("You have hypertension based on this reading.", "unsupported_diagnosis_assertion"),
        ("You should take this medication immediately.", "definitive_treatment_instruction"),
        ("Take 50 mg twice daily.", "definitive_treatment_instruction"),
        ("You should increase your dose right away.", "unsupported_dosage_change"),
        (
            "You don't need to see a doctor about this.",
            "unsafe_removal_of_clinician_involvement",
        ),
        ("This is not an emergency, so don't worry.", "unsupported_emergency_reassurance"),
    ],
)
def test_unsafe_directive_language_is_flagged_high_and_guarded(
    answer_text: str, expected_rule: str
) -> None:
    result = run_safety_checks(answer_text, disclaimer="Educational use only.")

    assert result.passed is False
    assert result.severity == SafetySeverity.HIGH
    assert any(flag.rule == expected_rule for flag in result.flags)
    assert result.guarded_response_required is True


def test_missing_disclaimer_is_flagged_low_but_not_guarded() -> None:
    result = run_safety_checks("General guideline information about blood pressure.", disclaimer="")

    assert result.passed is False
    assert result.severity == SafetySeverity.LOW
    assert any(flag.rule == "missing_disclaimer" for flag in result.flags)
    assert result.guarded_response_required is False


def test_whitespace_only_disclaimer_is_treated_as_missing() -> None:
    result = run_safety_checks("Some answer text.", disclaimer="   ")

    assert any(flag.rule == "missing_disclaimer" for flag in result.flags)


@pytest.mark.parametrize(
    "answer_text",
    [
        "Guidelines recommend a target blood pressure below 130/80 mmHg for most adults.",
        "According to the guideline, metformin is commonly used as a first-line therapy.",
        "General guideline information: regular monitoring is often recommended.",
        "Lifestyle changes such as diet and exercise are frequently discussed in guidelines.",
    ],
)
def test_neutral_educational_content_passes(answer_text: str) -> None:
    result = run_safety_checks(answer_text, disclaimer="Educational use only.")

    assert result.passed is True
    assert result.severity == SafetySeverity.NONE
    assert result.guarded_response_required is False
    assert result.flags == []


def test_multiple_flags_report_the_highest_severity() -> None:
    text = "You have hypertension. Take 50 mg twice daily."

    result = run_safety_checks(text, disclaimer="")

    rules = {flag.rule for flag in result.flags}
    assert "unsupported_diagnosis_assertion" in rules
    assert "definitive_treatment_instruction" in rules
    assert "missing_disclaimer" in rules
    assert result.severity == SafetySeverity.HIGH


def test_empty_answer_text_only_flags_disclaimer_if_missing() -> None:
    result = run_safety_checks("", disclaimer="Educational disclaimer.")

    assert result.passed is True
    assert result.flags == []
