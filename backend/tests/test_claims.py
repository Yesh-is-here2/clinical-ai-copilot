"""Unit tests for app.clinical.claims (Milestone E, section 10)."""

from __future__ import annotations

import pytest

from app.clinical.claims import Claim, DeterministicClaimExtractor, FakeClaimExtractor


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_splits_answer_into_one_claim_per_sentence() -> None:
    extractor = DeterministicClaimExtractor()
    text = "Guidelines recommend a target below 130/80 [C1]. Lifestyle changes help too [C2]."

    claims = await extractor.extract(text)

    assert len(claims) == 2
    assert claims[0].claim_id == "claim-1"
    assert claims[1].claim_id == "claim-2"


@pytest.mark.anyio
async def test_citation_markers_are_extracted_per_claim() -> None:
    extractor = DeterministicClaimExtractor()
    text = "The guideline supports this in two places [C1][C2]."

    claims = await extractor.extract(text)

    assert claims[0].cited_citation_ids == ["C1", "C2"]


@pytest.mark.anyio
async def test_sentence_with_no_citation_marker_is_still_a_claim() -> None:
    # Deliberate: an uncited factual sentence must survive as a claim so
    # downstream citation validation can flag it as unsupported, not be
    # silently dropped here.
    extractor = DeterministicClaimExtractor()
    text = "This is an uncited factual statement about treatment."

    claims = await extractor.extract(text)

    assert len(claims) == 1
    assert claims[0].cited_citation_ids == []


@pytest.mark.anyio
async def test_duplicate_citation_markers_are_deduplicated_in_order() -> None:
    extractor = DeterministicClaimExtractor()
    text = "Supported twice [C1][C1][C2]."

    claims = await extractor.extract(text)

    assert claims[0].cited_citation_ids == ["C1", "C2"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "sentence",
    [
        "This prototype is not a diagnostic tool.",
        "This system does not replace professional clinical judgment.",
        "This information is for educational and research purposes only.",
        "Please consult a doctor for personalized advice.",
        "Speak with your physician before making any changes.",
    ],
)
async def test_disclaimer_and_meta_sentences_are_excluded(sentence: str) -> None:
    extractor = DeterministicClaimExtractor()

    claims = await extractor.extract(sentence)

    assert claims == []


@pytest.mark.anyio
async def test_mixed_answer_keeps_claims_and_drops_disclaimers() -> None:
    extractor = DeterministicClaimExtractor()
    text = (
        "Guidelines recommend regular monitoring [C1]. "
        "This system does not replace professional clinical judgment. "
        "Lifestyle changes are often beneficial [C2]."
    )

    claims = await extractor.extract(text)

    assert len(claims) == 2
    assert all("does not replace" not in c.text for c in claims)


@pytest.mark.anyio
async def test_blank_answer_produces_no_claims() -> None:
    extractor = DeterministicClaimExtractor()

    assert await extractor.extract("") == []
    assert await extractor.extract("   ") == []


@pytest.mark.anyio
async def test_fake_claim_extractor_returns_fixed_claims_regardless_of_text() -> None:
    fixed = [Claim(claim_id="claim-1", text="fixed claim", cited_citation_ids=["C1"])]
    extractor = FakeClaimExtractor(fixed)

    result_a = await extractor.extract("anything")
    result_b = await extractor.extract("something completely different")

    assert result_a == fixed
    assert result_b == fixed
