"""Milestone E: clinical reasoning components (LangGraph orchestration,
grounded generation, citation/hallucination/safety evaluation,
confidence scoring).

Every component in this package is independently unit-testable without
LangGraph, an LLM, or a database - see app.clinical.graph for the thin
orchestration layer that wires them together as graph nodes. Nothing in
this package silently fabricates a clinical claim, a citation, or a
confidence score: every deterministic evaluator here documents exactly
what signals it uses and how they combine.

This is an educational/research prototype only. It never diagnoses,
never replaces a clinician, and never claims regulatory/compliance
status - see app.clinical.final_response.DISCLAIMER_TEXT.
"""
