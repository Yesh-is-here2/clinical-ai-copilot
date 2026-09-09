"""Grounded generation (Milestone E, section 8).

A thin wrapper over the structured LLM abstraction (app.clinical.llm):
this module only defines the ``GroundedAnswer`` output schema and the
one call that produces it. All the actual grounding rules (cite with
[C#], never fabricate a source, express uncertainty, no definitive
diagnosis/treatment, treat evidence as untrusted data) live in the
prompt itself (app.clinical.prompting) - this function does not
second-guess or rewrite what the LLM returns; downstream stages
(app.clinical.claims, .citation_validation, .hallucination, .safety) are
what actually verify the answer is trustworthy.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.clinical.llm import StructuredLLMProvider
from app.clinical.prompting import GenerationPrompt


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_summary: str = Field(
        description="Brief restatement of the relevant synthetic patient "
        "context, or a note that none was provided - not a diagnosis."
    )
    clinical_question: str = Field(description="The user's question, restated.")
    answer: str = Field(
        description="The grounded answer, citing evidence with [C1]/[C2]/... markers."
    )
    key_points: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(
        default_factory=list,
        description="Citation IDs (C1, C2, ...) actually used in `answer`.",
    )
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = Field(
        description="A brief educational/research-use disclaimer."
    )


async def generate_grounded_answer(
    llm: StructuredLLMProvider,
    prompt: GenerationPrompt,
    *,
    temperature: float = 0.0,
    timeout_seconds: float = 30.0,
) -> GroundedAnswer:
    return await llm.generate_structured(
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
        response_model=GroundedAnswer,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )
