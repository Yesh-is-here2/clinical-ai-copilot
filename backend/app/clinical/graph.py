"""LangGraph orchestration for the clinical query pipeline (Milestone E,
sections 2/3).

This module is intentionally thin: every node is a few lines that calls
into an already independently-tested function from elsewhere in
``app.clinical``/``app.retrieval`` and shapes the result into a state
update. No clinical reasoning/business logic is embedded in the graph
itself - see the module docstrings across ``app.clinical`` for where
each decision actually lives.

Routes (section 3):
    prompt injection detected      -> blocked (safe message, no evidence)
    both retrieval sources fail    -> guarded (safe message, no evidence)
    insufficient evidence          -> abstain (no generation attempted)
    LLM/generation failure         -> guarded (safe message)
    citation validation failure    -> regenerate once, then continue regardless
    hallucination above threshold  -> guarded final answer text
    safety check requires guarding -> guarded final answer text
    still-unsupported after regen  -> guarded final answer text (backstop)
    otherwise                      -> normal structured response
Every one of the above then always continues:
    construct_final_response -> write_audit_log -> END

Citation validation and hallucination evaluation are related but
deliberately distinct safety layers (they are not folded into one
check): citation validation asks "does this claim correctly cite
supporting evidence?" (app.clinical.citation_validation); hallucination
evaluation asks "is the claim itself supported, unsupported,
contradicted, or uncertain relative to the evidence?"
(app.clinical.hallucination). Citation validation still drives the (at
most one) regeneration attempt, but once that budget is exhausted,
``_route_after_citation_validation`` always continues on to
hallucination detection and safety checks rather than short-circuiting -
so a genuinely contradicted or high-hallucination-rate answer can still
independently earn its own, more specific "guarded_hallucination"/
"guarded_safety" terminal reason. Citation enforcement is never
weakened by this: ``construct_final_response_node``'s
``_determine_terminal_reason`` backstops with "guarded_citation_failure"
if nothing more specific already fired and important unsupported claims
remain - so a citation failure with a budget-exhausted regeneration
attempt is always guarded one way or another, just possibly with a more
informative reason than the citation layer alone would have given it.

Regeneration is capped at ``ClinicalGraphDependencies.max_regeneration_
attempts`` (default 1, section 3/11) - ``regeneration_count`` on the
state and the routing function below make a second, unbounded loop
impossible regardless of how many claims are still unsupported.

Audit logging (section 2) is its own real graph node
(``write_audit_log``), not a side effect folded into
``construct_final_response`` - see that node's docstring for why, and
for the explicit, non-propagating failure behavior when the injected
``AuditLogWriter`` itself raises.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import ValidationError as PydanticValidationError

from langgraph.graph import END, START, StateGraph

from app.clinical.audit import AuditLogEntry, AuditLogWriter, NullAuditLogWriter
from app.clinical.citation_validation import (
    CitationValidationReport,
    CitationValidationStatus,
    validate_citations,
)
from app.clinical.citations import Citation, build_citations, citations_by_id, filter_valid_citation_ids
from app.clinical.claims import ClaimExtractor, DeterministicClaimExtractor
from app.clinical.classification import DeterministicQuestionClassifier, QuestionClassifier
from app.clinical.confidence import ConfidenceConfig, ConfidenceLabel, calculate_confidence
from app.clinical.evidence_sufficiency import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    EvidenceSufficiencyConfig,
    EvidenceSufficiencyStatus,
    evaluate_evidence_sufficiency,
)
from app.clinical.final_response import DISCLAIMER_TEXT, FinalClinicalResponse
from app.clinical.generation import generate_grounded_answer
from app.clinical.hallucination import HallucinationEvaluation, evaluate_hallucinations
from app.clinical.injection import detect_prompt_injection
from app.clinical.llm import LLMError, StructuredLLMProvider
from app.clinical.prompting import build_generation_prompt, build_regeneration_prompt
from app.clinical.safety import SafetyCheckResult, SafetySeverity, run_safety_checks
from app.clinical.state import ClinicalState
from app.retrieval.hybrid import HybridRetrievalError, HybridRetrievalService, RetrievalComponentStatus
from app.schemas.patient import PatientContext


def evidence_sufficiency_config_from_settings(settings) -> EvidenceSufficiencyConfig:
    """Pure mapping from ``app.core.config.Settings`` to
    ``EvidenceSufficiencyConfig`` - not called anywhere in this module
    itself (this file stays free of a ``Settings`` import at module
    scope beyond this one function's parameter), but available for
    whatever call site (a future API route, a CLI) constructs
    ``ClinicalGraphDependencies`` for real, so it does not have to
    hand-copy every threshold field name."""

    return EvidenceSufficiencyConfig(
        sufficient_threshold=settings.evidence_sufficient_threshold,
        partial_threshold=settings.evidence_partial_threshold,
        target_evidence_count=settings.evidence_target_count,
        target_independent_source_count=settings.evidence_target_source_count,
    )


def confidence_config_from_settings(settings) -> ConfidenceConfig:
    """Pure mapping from ``app.core.config.Settings`` to
    ``ConfidenceConfig`` - see ``evidence_sufficiency_config_from_settings``
    for why this lives here rather than being wired implicitly."""

    return ConfidenceConfig(
        high_threshold=settings.confidence_high_threshold,
        moderate_threshold=settings.confidence_moderate_threshold,
        severe_hallucination_rate_threshold=settings.severe_hallucination_rate_threshold,
    )


@dataclass
class ClinicalGraphDependencies:
    """Everything the graph needs injected. Construct real dependencies
    (HybridRetrievalService, OpenAIStructuredLLMProvider, ...) at the
    call site (e.g. FastAPI startup / a route handler) - never inside
    this module, which must stay import-safe with no network/DB/model
    access."""

    hybrid_retrieval: HybridRetrievalService
    llm_provider: StructuredLLMProvider
    question_classifier: QuestionClassifier = field(
        default_factory=DeterministicQuestionClassifier
    )
    claim_extractor: ClaimExtractor = field(default_factory=DeterministicClaimExtractor)
    audit_writer: AuditLogWriter = field(default_factory=NullAuditLogWriter)
    evidence_sufficiency_config: EvidenceSufficiencyConfig = field(
        default_factory=EvidenceSufficiencyConfig
    )
    confidence_config: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    generation_temperature: float = 0.0
    generation_timeout_seconds: float = 30.0
    max_regeneration_attempts: int = 1


def _record_timing(state: ClinicalState, stage: str, elapsed_seconds: float) -> dict[str, float]:
    timings = dict(state.get("stage_timings", {}))
    timings[stage] = elapsed_seconds
    return timings


def _used_citation_ids(report: CitationValidationReport | None) -> set[str]:
    if report is None:
        return set()

    used: set[str] = set()
    for validation in report.claim_validations:
        if validation.status in (
            CitationValidationStatus.SUPPORTED,
            CitationValidationStatus.PARTIALLY_SUPPORTED,
        ):
            used.update(validation.supporting_citation_ids)

    return used


def _default_patient_summary(patient_context: PatientContext | None) -> str:
    if patient_context is None:
        return "No patient context was provided for this question."

    return f"Synthetic patient {patient_context.patient_id}."


# --- Routing (section 3) --------------------------------------------


def _route_after_injection(state: ClinicalState) -> str:
    return "blocked" if state.get("terminal_reason") == "blocked_injection" else "continue"


def _route_after_retrieval(state: ClinicalState) -> str:
    return "failed" if state.get("terminal_reason") == "error_retrieval_failed" else "continue"


def _route_after_sufficiency(state: ClinicalState) -> str:
    return (
        "abstain"
        if state.get("terminal_reason") == "abstained_insufficient_evidence"
        else "continue"
    )


def _route_after_generation(state: ClinicalState) -> str:
    return "failed" if state.get("terminal_reason") == "error_generation_failed" else "continue"


def build_clinical_graph(deps: ClinicalGraphDependencies):
    def _route_after_citation_validation(state: ClinicalState) -> str:
        # A closure (unlike the other, deps-free routing functions above)
        # because the regeneration budget lives on `deps`, not on the
        # state - regeneration_count is state, max_regeneration_attempts
        # is configuration. Comparing them directly here, rather than via
        # a terminal_reason sentinel set by validate_citations_node,
        # means validate_citations_node itself never decides anything
        # terminal: once the budget is exhausted this simply falls
        # through to "continue" so hallucination detection and safety
        # checks still run on the final (possibly still-unsupported)
        # answer - see construct_final_response_node for where a
        # still-unsupported claim, if nothing more specific (hallucination
        # /safety) already claimed the terminal reason by then, finally
        # becomes a guarded response. This is also what makes a second,
        # unbounded regeneration loop structurally impossible: regardless
        # of the report, once regeneration_count reaches the budget this
        # always returns "continue", never "regenerate", again.
        report = state.get("citation_validation")
        regeneration_count = state.get("regeneration_count", 0)

        if (
            report is not None
            and report.has_important_unsupported_claims
            and regeneration_count < deps.max_regeneration_attempts
        ):
            return "regenerate"

        return "continue"

    async def validate_input(state: ClinicalState) -> dict:
        start = time.perf_counter()
        question = (state.get("question") or "").strip()
        component_errors = list(state.get("component_errors", []))

        if not question:
            component_errors.append("validate_input: question is blank.")

        return {
            "question": question,
            "warnings": list(state.get("warnings", [])),
            "component_errors": component_errors,
            "regeneration_count": state.get("regeneration_count", 0),
            "terminal_reason": state.get("terminal_reason"),
            "stage_timings": _record_timing(state, "validate_input", time.perf_counter() - start),
        }

    async def parse_patient_context(state: ClinicalState) -> dict:
        start = time.perf_counter()
        raw = state.get("patient_context")
        warnings = list(state.get("warnings", []))
        component_errors = list(state.get("component_errors", []))

        if raw is None or isinstance(raw, PatientContext):
            parsed = raw
        else:
            try:
                parsed = PatientContext.model_validate(raw)
            except PydanticValidationError as exc:
                parsed = None
                warnings.append(
                    "Provided patient context was malformed and was ignored."
                )
                component_errors.append(f"parse_patient_context: {exc}")

        return {
            "patient_context": parsed,
            "warnings": warnings,
            "component_errors": component_errors,
            "stage_timings": _record_timing(
                state, "parse_patient_context", time.perf_counter() - start
            ),
        }

    async def classify_question(state: ClinicalState) -> dict:
        start = time.perf_counter()
        result = await deps.question_classifier.classify(state.get("question", ""))

        return {
            "question_category": result.category,
            "stage_timings": _record_timing(
                state, "classification", time.perf_counter() - start
            ),
        }

    async def detect_prompt_injection_node(state: ClinicalState) -> dict:
        start = time.perf_counter()
        result = detect_prompt_injection(state.get("question", ""))

        updates: dict = {
            "injection_result": result,
            "stage_timings": _record_timing(
                state, "injection_detection", time.perf_counter() - start
            ),
        }
        if result.detected:
            updates["terminal_reason"] = "blocked_injection"

        return updates

    async def build_search_query(state: ClinicalState) -> dict:
        # Deliberately just the raw question: hybrid retrieval (BM25 +
        # vector + RRF + rerank, Milestone D) already handles lexical
        # and semantic matching without needing query expansion here,
        # and this avoids inventing search terms not actually present
        # in what the user asked.
        return {"search_query": state.get("question", "")}

    async def retrieve_evidence(state: ClinicalState) -> dict:
        start = time.perf_counter()
        query = state.get("search_query") or state.get("question", "")
        warnings = list(state.get("warnings", []))
        component_errors = list(state.get("component_errors", []))

        try:
            result = await deps.hybrid_retrieval.retrieve(query)
        except HybridRetrievalError as exc:
            component_errors.append(f"retrieve_evidence: {exc}")
            return {
                "terminal_reason": "error_retrieval_failed",
                "retrieved_evidence": [],
                "component_errors": component_errors,
                "warnings": warnings,
                "stage_timings": _record_timing(
                    state, "retrieval", time.perf_counter() - start
                ),
            }

        if result.degraded:
            if result.vector.status is RetrievalComponentStatus.FAILED:
                warnings.append(f"Vector retrieval degraded: {result.vector.error_message}")
            if result.bm25.status is RetrievalComponentStatus.FAILED:
                warnings.append(f"BM25 retrieval degraded: {result.bm25.error_message}")

        return {
            "retrieval_result": result,
            "retrieved_evidence": result.evidence,
            "warnings": warnings,
            "stage_timings": _record_timing(state, "retrieval", time.perf_counter() - start),
        }

    async def rerank_evidence(state: ClinicalState) -> dict:
        # HybridRetrievalService.retrieve() already reranks internally
        # (Milestone D) - re-deriving that here would duplicate Phase 4
        # internals, which is explicitly out of scope. This node's job
        # is (a) assigning deterministic citation IDs from the final
        # evidence order, and (b) propagating reranker degradation.
        start = time.perf_counter()
        evidence = state.get("retrieved_evidence", [])
        retrieval_result = state.get("retrieval_result")
        warnings = list(state.get("warnings", []))

        if (
            retrieval_result is not None
            and retrieval_result.reranker.status is RetrievalComponentStatus.FAILED
        ):
            warnings.append(f"Reranking degraded: {retrieval_result.reranker.error_message}")

        citations = build_citations(evidence)

        return {
            "citations": citations,
            "warnings": warnings,
            "stage_timings": _record_timing(
                state, "rerank_evidence", time.perf_counter() - start
            ),
        }

    async def evaluate_evidence_sufficiency_node(state: ClinicalState) -> dict:
        start = time.perf_counter()
        evidence = state.get("retrieved_evidence", [])
        result = evaluate_evidence_sufficiency(
            evidence, config=deps.evidence_sufficiency_config
        )

        updates: dict = {
            "evidence_sufficiency": result,
            "stage_timings": _record_timing(
                state, "evidence_sufficiency", time.perf_counter() - start
            ),
        }
        if result.status is EvidenceSufficiencyStatus.INSUFFICIENT:
            updates["terminal_reason"] = "abstained_insufficient_evidence"

        return updates

    async def generate_grounded_answer_node(state: ClinicalState) -> dict:
        start = time.perf_counter()
        component_errors = list(state.get("component_errors", []))
        warnings = list(state.get("warnings", []))
        citations = state.get("citations", [])
        prior_report = state.get("citation_validation")

        is_regeneration = bool(prior_report and prior_report.has_important_unsupported_claims)
        regeneration_count = state.get("regeneration_count", 0) + (1 if is_regeneration else 0)

        base_prompt = build_generation_prompt(
            question=state.get("question", ""),
            patient_context=state.get("patient_context"),
            citations=citations,
        )

        if is_regeneration:
            claims_by_id = {claim.claim_id: claim for claim in state.get("extracted_claims", [])}
            unsupported_texts = [
                claims_by_id[validation.claim_id].text
                for validation in prior_report.claim_validations
                if validation.claim_id in claims_by_id
                and validation.status is CitationValidationStatus.UNSUPPORTED
            ]
            prompt = build_regeneration_prompt(base_prompt, unsupported_texts)
        else:
            prompt = base_prompt

        try:
            answer = await generate_grounded_answer(
                deps.llm_provider,
                prompt,
                temperature=deps.generation_temperature,
                timeout_seconds=deps.generation_timeout_seconds,
            )
        except LLMError as exc:
            component_errors.append(f"generate_grounded_answer: {exc}")
            return {
                "terminal_reason": "error_generation_failed",
                "component_errors": component_errors,
                "regeneration_count": regeneration_count,
                "stage_timings": _record_timing(
                    state, "generation", time.perf_counter() - start
                ),
            }

        # Never let a fabricated citation ID survive, even at the
        # top-level GroundedAnswer.citation_ids the LLM reported using -
        # per-claim filtering happens again in validate_citations, but
        # this list is user-visible metadata in its own right.
        known = citations_by_id(citations)
        valid_ids, fabricated_ids = filter_valid_citation_ids(answer.citation_ids, known)
        if fabricated_ids:
            answer = answer.model_copy(update={"citation_ids": valid_ids})
            warnings.append(
                f"Generated response referenced fabricated citation ID(s) "
                f"{fabricated_ids}; removed."
            )

        return {
            "generated_response": answer,
            "warnings": warnings,
            "component_errors": component_errors,
            "regeneration_count": regeneration_count,
            "stage_timings": _record_timing(state, "generation", time.perf_counter() - start),
        }

    async def extract_claims_node(state: ClinicalState) -> dict:
        start = time.perf_counter()
        answer = state.get("generated_response")
        claims = await deps.claim_extractor.extract(answer.answer if answer else "")

        return {
            "extracted_claims": claims,
            "stage_timings": _record_timing(
                state, "claim_extraction", time.perf_counter() - start
            ),
        }

    async def validate_citations_node(state: ClinicalState) -> dict:
        # Deliberately decides nothing terminal - just computes and
        # returns the report. Whether that report's still-unsupported
        # claims (after the regeneration budget is exhausted) end up
        # guarding the final response is decided later, in
        # construct_final_response_node, *after* hallucination detection
        # and safety checks have had their own, more specific chance to
        # explain why (see the module docstring's route table).
        start = time.perf_counter()
        claims = state.get("extracted_claims", [])
        citations = state.get("citations", [])
        evidence = state.get("retrieved_evidence", [])

        report = validate_citations(claims, citations, evidence)

        return {
            "citation_validation": report,
            "stage_timings": _record_timing(
                state, "citation_validation", time.perf_counter() - start
            ),
        }

    async def detect_hallucinations_node(state: ClinicalState) -> dict:
        start = time.perf_counter()
        claims = state.get("extracted_claims", [])
        report = state.get("citation_validation") or CitationValidationReport()
        evidence = state.get("retrieved_evidence", [])

        evaluation = evaluate_hallucinations(claims, report, evidence)

        updates: dict = {
            "hallucination_evaluation": evaluation,
            "stage_timings": _record_timing(
                state, "hallucination_detection", time.perf_counter() - start
            ),
        }
        if (
            evaluation.hallucination_rate
            >= deps.confidence_config.severe_hallucination_rate_threshold
            and state.get("terminal_reason") is None
        ):
            updates["terminal_reason"] = "guarded_hallucination"

        return updates

    async def run_safety_checks_node(state: ClinicalState) -> dict:
        start = time.perf_counter()
        answer = state.get("generated_response")
        result = run_safety_checks(
            answer.answer if answer else "",
            disclaimer=answer.disclaimer if answer else "",
        )

        updates: dict = {
            "safety_evaluation": result,
            "stage_timings": _record_timing(state, "safety", time.perf_counter() - start),
        }
        if result.guarded_response_required and state.get("terminal_reason") is None:
            updates["terminal_reason"] = "guarded_safety"

        return updates

    async def calculate_confidence_node(state: ClinicalState) -> dict:
        start = time.perf_counter()
        evidence_sufficiency = state.get("evidence_sufficiency") or evaluate_evidence_sufficiency(
            [], config=deps.evidence_sufficiency_config
        )
        citation_validation = state.get("citation_validation") or CitationValidationReport()
        hallucination_evaluation = state.get("hallucination_evaluation") or HallucinationEvaluation()
        safety_evaluation = state.get("safety_evaluation") or SafetyCheckResult(
            passed=True, severity=SafetySeverity.NONE
        )
        retrieval_result = state.get("retrieval_result")
        degraded = retrieval_result.degraded if retrieval_result is not None else False

        result = calculate_confidence(
            evidence_sufficiency=evidence_sufficiency,
            citation_validation=citation_validation,
            hallucination_evaluation=hallucination_evaluation,
            safety_result=safety_evaluation,
            retrieval_degraded=degraded,
            config=deps.confidence_config,
        )

        return {
            "confidence": result,
            "stage_timings": _record_timing(state, "confidence", time.perf_counter() - start),
        }

    def _determine_terminal_reason(state: ClinicalState) -> str:
        # Runs only here, once every upstream stage (injection, retrieval,
        # sufficiency, generation, hallucination, safety) has already had
        # its chance to claim a more specific terminal_reason. The one
        # case nothing upstream necessarily resolves is: citation
        # validation still has important unsupported claims *after* the
        # regeneration budget is exhausted (see
        # _route_after_citation_validation), but hallucination detection
        # and safety checks both happened to pass. Citation enforcement
        # must never be weakened just because those other two layers
        # didn't independently object - so this is the guaranteed final
        # backstop, not the primary detection path for that case.
        terminal_reason = state.get("terminal_reason")
        if terminal_reason is not None:
            return terminal_reason

        citation_validation = state.get("citation_validation")
        regeneration_count = state.get("regeneration_count", 0)
        if (
            citation_validation is not None
            and citation_validation.has_important_unsupported_claims
            and regeneration_count >= deps.max_regeneration_attempts
        ):
            return "guarded_citation_failure"

        return "normal"

    async def construct_final_response_node(state: ClinicalState) -> dict:
        start = time.perf_counter()
        request_id = state.get("request_id", "")
        question = state.get("question", "")
        patient_context = state.get("patient_context")
        terminal_reason = _determine_terminal_reason(state)
        citations = state.get("citations", [])
        citation_validation = state.get("citation_validation")
        safety_evaluation = state.get("safety_evaluation")
        confidence_result = state.get("confidence")
        generated = state.get("generated_response")
        evidence_sufficiency = state.get("evidence_sufficiency")
        warnings = list(state.get("warnings", []))

        patient_summary = (
            generated.patient_summary if generated else _default_patient_summary(patient_context)
        )
        key_points = generated.key_points if generated else []
        safety_flags = [flag.rule for flag in safety_evaluation.flags] if safety_evaluation else []

        final_citations: list[Citation]

        if terminal_reason == "blocked_injection":
            injection_result = state.get("injection_result")
            answer = (
                injection_result.safe_message
                if injection_result and injection_result.safe_message
                else "This request could not be processed."
            )
            final_citations = []
            warnings.append("Request blocked: potential prompt injection detected.")

        elif terminal_reason == "error_retrieval_failed":
            answer = (
                "This question could not be answered because evidence "
                f"retrieval failed. {INSUFFICIENT_EVIDENCE_MESSAGE}"
            )
            final_citations = []
            warnings.append("Evidence retrieval failed for this request.")

        elif terminal_reason == "abstained_insufficient_evidence":
            answer = INSUFFICIENT_EVIDENCE_MESSAGE
            final_citations = citations[:3]  # closest evidence, not presented as an answer

        elif terminal_reason == "error_generation_failed":
            answer = (
                "This question could not be answered because answer "
                f"generation was unavailable. {INSUFFICIENT_EVIDENCE_MESSAGE}"
            )
            final_citations = citations[:3]
            warnings.append("Answer generation failed or was unavailable.")

        elif terminal_reason in (
            "guarded_citation_failure",
            "guarded_hallucination",
            "guarded_safety",
        ):
            answer = (
                "This system could not fully verify part of its answer "
                "against the retrieved evidence, so the unverified "
                f"content has been withheld. {INSUFFICIENT_EVIDENCE_MESSAGE}"
            )
            used_ids = _used_citation_ids(citation_validation)
            final_citations = [c for c in citations if c.citation_id in used_ids]
            warnings.append(f"Response guarded due to: {terminal_reason}.")

        else:
            answer = generated.answer if generated else INSUFFICIENT_EVIDENCE_MESSAGE
            used_ids = _used_citation_ids(citation_validation)
            final_citations = (
                [c for c in citations if c.citation_id in used_ids] if used_ids else citations
            )

        evidence_status = (
            evidence_sufficiency.status
            if evidence_sufficiency
            else EvidenceSufficiencyStatus.INSUFFICIENT
        )
        confidence_score = confidence_result.score if confidence_result else 0.0
        confidence_label = confidence_result.label if confidence_result else ConfidenceLabel.LOW

        final_response = FinalClinicalResponse(
            request_id=request_id,
            patient_summary=patient_summary,
            clinical_question=question,
            answer=answer,
            key_points=key_points,
            citations=final_citations,
            confidence=confidence_score,
            confidence_label=confidence_label,
            evidence_status=evidence_status,
            warnings=warnings,
            safety_flags=safety_flags,
            disclaimer=DISCLAIMER_TEXT,
        )

        own_elapsed = time.perf_counter() - start
        total_elapsed = sum(state.get("stage_timings", {}).values()) + own_elapsed

        return {
            "final_response": final_response,
            "terminal_reason": terminal_reason,
            "warnings": warnings,
            "stage_timings": _record_timing(state, "final_response", own_elapsed),
            "total_elapsed_seconds": total_elapsed,
        }

    async def write_audit_log_node(state: ClinicalState) -> dict:
        # A real graph node (not a call folded into construct_final_
        # response_node) so it is independently visible in stage_timings,
        # independently routable, and - most importantly - so a failure
        # writing the audit record can never affect the already-built
        # FinalClinicalResponse (see the try/except below): audit
        # persistence is an observability concern, not a clinical-safety
        # one, and must never be allowed to turn a safe, already-decided
        # response into a failed request. Runs on every terminal path
        # (injection block, retrieval/generation failure, abstention,
        # guarded, and normal) because every path passes through
        # construct_final_response first, and this is its only successor.
        #
        # Only structured, audit-safe fields go on the entry - see
        # app.clinical.audit.AuditLogEntry's own docstring for the "never
        # store system prompts or chain-of-thought" boundary. Everything
        # here is either a plain identifier, a score, a status string, or
        # a short warning/reason string already produced by an earlier
        # stage - never raw evidence text, the system prompt, or an LLM's
        # raw response.
        start = time.perf_counter()
        request_id = state.get("request_id", "")
        question = state.get("question", "")
        patient_context = state.get("patient_context")
        final_response = state.get("final_response")
        terminal_reason = state.get("terminal_reason") or "unknown"
        question_category = state.get("question_category")
        citation_validation = state.get("citation_validation")
        hallucination_evaluation = state.get("hallucination_evaluation")
        confidence_result = state.get("confidence")
        component_errors = list(state.get("component_errors", []))

        entry = AuditLogEntry(
            request_id=request_id,
            patient_id=patient_context.patient_id if isinstance(patient_context, PatientContext) else None,
            question=question,
            question_category=question_category.value if question_category else None,
            model=deps.llm_provider.model,
            retrieved_document_ids=sorted(
                {item.document_id for item in state.get("retrieved_evidence", [])}
            ),
            output_status=terminal_reason,
            evidence_sufficiency=(
                final_response.evidence_status.value if final_response else None
            ),
            confidence=final_response.confidence if final_response else None,
            confidence_label=(
                final_response.confidence_label.value if final_response else None
            ),
            latency_ms=(
                state.get("total_elapsed_seconds") * 1000
                if state.get("total_elapsed_seconds") is not None
                else None
            ),
            safety_flags=final_response.safety_flags if final_response else [],
            hallucination_rate=(
                hallucination_evaluation.hallucination_rate
                if hallucination_evaluation
                else None
            ),
            citation_correctness=(
                citation_validation.overall_correctness_score if citation_validation else None
            ),
            warnings=final_response.warnings if final_response else [],
            reasons=confidence_result.reasons if confidence_result else [],
            stage_timings=dict(state.get("stage_timings", {})),
            actor_user_id=state.get("actor_user_id"),
            actor_role=state.get("actor_role"),
        )

        try:
            await deps.audit_writer.write(entry)
        except Exception as exc:
            # Explicit, safe failure behavior (Phase 5 constraint: audit
            # write failure must never cause unsupported clinical
            # generation): record the failure for observability, but
            # never raise - the already-constructed, already-safe
            # final_response from construct_final_response_node is still
            # returned to the caller unchanged.
            component_errors.append(f"write_audit_log: {exc}")

        return {
            "component_errors": component_errors,
            "stage_timings": _record_timing(state, "audit", time.perf_counter() - start),
        }

    graph = StateGraph(ClinicalState)

    graph.add_node("validate_input", validate_input)
    graph.add_node("parse_patient_context", parse_patient_context)
    graph.add_node("classify_question", classify_question)
    graph.add_node("detect_prompt_injection", detect_prompt_injection_node)
    graph.add_node("build_search_query", build_search_query)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("rerank_evidence", rerank_evidence)
    graph.add_node("evaluate_evidence_sufficiency", evaluate_evidence_sufficiency_node)
    graph.add_node("generate_grounded_answer", generate_grounded_answer_node)
    graph.add_node("extract_claims", extract_claims_node)
    graph.add_node("validate_citations", validate_citations_node)
    graph.add_node("detect_hallucinations", detect_hallucinations_node)
    graph.add_node("run_safety_checks", run_safety_checks_node)
    graph.add_node("calculate_confidence", calculate_confidence_node)
    graph.add_node("construct_final_response", construct_final_response_node)
    graph.add_node("write_audit_log", write_audit_log_node)

    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "parse_patient_context")
    graph.add_edge("parse_patient_context", "classify_question")
    graph.add_edge("classify_question", "detect_prompt_injection")

    graph.add_conditional_edges(
        "detect_prompt_injection",
        _route_after_injection,
        {"blocked": "construct_final_response", "continue": "build_search_query"},
    )
    graph.add_edge("build_search_query", "retrieve_evidence")
    graph.add_conditional_edges(
        "retrieve_evidence",
        _route_after_retrieval,
        {"failed": "construct_final_response", "continue": "rerank_evidence"},
    )
    graph.add_edge("rerank_evidence", "evaluate_evidence_sufficiency")
    graph.add_conditional_edges(
        "evaluate_evidence_sufficiency",
        _route_after_sufficiency,
        {"abstain": "calculate_confidence", "continue": "generate_grounded_answer"},
    )
    graph.add_conditional_edges(
        "generate_grounded_answer",
        _route_after_generation,
        {"failed": "construct_final_response", "continue": "extract_claims"},
    )
    graph.add_edge("extract_claims", "validate_citations")
    graph.add_conditional_edges(
        "validate_citations",
        _route_after_citation_validation,
        {"regenerate": "generate_grounded_answer", "continue": "detect_hallucinations"},
    )
    graph.add_edge("detect_hallucinations", "run_safety_checks")
    graph.add_edge("run_safety_checks", "calculate_confidence")
    graph.add_edge("calculate_confidence", "construct_final_response")
    graph.add_edge("construct_final_response", "write_audit_log")
    graph.add_edge("write_audit_log", END)

    return graph.compile()


async def run_clinical_query(
    graph,
    *,
    request_id: str,
    question: str,
    patient_context: PatientContext | dict | None = None,
) -> FinalClinicalResponse:
    """Convenience entrypoint: build the initial state, invoke the
    compiled graph, and return just the final response."""

    initial_state: ClinicalState = {
        "request_id": request_id,
        "question": question,
        "patient_context": patient_context,
        "warnings": [],
        "component_errors": [],
        "regeneration_count": 0,
        "terminal_reason": None,
        "stage_timings": {},
    }

    result_state = await graph.ainvoke(initial_state)

    return result_state["final_response"]
