"""Unit tests for Phase 6 claim-level faithfulness and hallucination detection."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from src.evaluation.semantic_similarity import BaseEmbeddingProvider
from src.generation import GenerationResult
from src.hallucination import (
    LABEL_CONTRADICTED,
    LABEL_NOT_ENOUGH_INFO,
    LABEL_SUPPORTED,
    BaseNLIScorer,
    ClaimExtractor,
    HallucinationDetectionRecord,
    HallucinationDetector,
    NLIScores,
    load_hallucination_results,
    save_hallucination_results,
)


class MockTopicEmbedder(BaseEmbeddingProvider):
    """Deterministic keyword-cluster embedder for offline retrieval/verification tests."""

    def __init__(self) -> None:
        self.model_name = "mock-all-MiniLM-L6-v2"

    def _vec(self, text: str) -> np.ndarray:
        low = text.lower()
        if any(k in low for k in ("meridian", "satellite", "phase 2", "radar", "orbit")):
            return np.array([0.95, 0.1, 0.0, 0.0], dtype=np.float64)
        if any(k in low for k in ("focaccia", "olive", "dough", "rosemary")):
            return np.array([0.0, 0.0, 0.95, 0.1], dtype=np.float64)
        if any(k in low for k in ("titanium", "melting", "kestrel", "bridge", "cost")):
            return np.array([0.0, 0.0, 0.1, 0.95], dtype=np.float64)
        if any(k in low for k in ("water", "boils", "france", "lyon", "paris")):
            return np.array([0.0, 0.95, 0.1, 0.0], dtype=np.float64)
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])


class MockNLIScorer(BaseNLIScorer):
    """Mock NLI cross-encoder driven by explicit test rules."""

    def __init__(self) -> None:
        self.model_name = "mock-nli-deberta-v3-small"

    def score(self, premise: str, hypothesis: str) -> NLIScores:
        p_low = premise.lower()
        h_low = hypothesis.lower()
        if "72" in p_low and "72" in h_low and "water" in h_low:
            return NLIScores(entailment=0.91, contradiction=0.04, neutral=0.05)
        if "72" in p_low and "100" in h_low and "water" in h_low:
            return NLIScores(entailment=0.03, contradiction=0.92, neutral=0.05)
        if "lyon" in p_low and "lyon" in h_low:
            return NLIScores(entailment=0.92, contradiction=0.03, neutral=0.05)
        if "lyon" in p_low and "paris" in h_low:
            return NLIScores(entailment=0.02, contradiction=0.93, neutral=0.05)
        if "phase 2" in p_low and "18" in h_low and "radar" in h_low:
            return NLIScores(entailment=0.90, contradiction=0.04, neutral=0.06)
        return NLIScores(entailment=0.20, contradiction=0.10, neutral=0.70)


def test_claim_extraction_multiple_claims_and_filters() -> None:
    """Verify compound claim decomposition, typing, filler filtering, and refusals."""
    extractor = ClaimExtractor()
    res = extractor.extract(
        question_id="Q066",
        model_id="mock-model",
        response_text=(
            "Sure! Phase 2 added 18 radar satellites in 2023, and Phase 3 completed "
            "the network with 6 units. Hope this helps."
        ),
    )
    assert res.status == "extracted"
    assert len(res.claims) == 2
    assert res.claims[0].claim_id == "Q066_C01"
    assert res.claims[0].claim_type == "numeric_claim"
    assert res.claims[1].claim_id == "Q066_C02"
    assert len(res.non_factual_segments) == 2

    # Pure refusal response yields 0 factual claims with refusal_only status
    refusal = extractor.extract(
        question_id="Q074",
        model_id="mock-model",
        response_text="The provided context does not state the construction cost of the bridge.",
    )
    assert refusal.status == "refusal_only"
    assert len(refusal.claims) == 0
    assert len(refusal.refusal_segments) == 1


def test_fully_supported_and_fully_contradicted_answers() -> None:
    """Verify SUPPORTED vs CONTRADICTED labeling and exact faithfulness ratios."""
    context = (
        "The Meridian constellation launched in three phases. "
        "Phase 2 added 18 synthetic-aperture radar satellites in 2023."
    )
    detector = HallucinationDetector(embedder=MockTopicEmbedder())

    supported = detector.detect_single(
        question_id="Q066",
        model_id="mock-a",
        generated_answer="Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
        context=context,
        category="Context-based Question Answering",
        hallucination_test_type="context_supported_precision",
    )
    assert supported.faithfulness.supported_count == 1
    assert supported.faithfulness.faithfulness_ratio == pytest.approx(1.0)
    assert supported.faithfulness.contradiction_rate == pytest.approx(0.0)
    assert supported.hallucination_detected is False
    assert supported.claims[0].verification_label == LABEL_SUPPORTED

    # Numerical contradiction (24 vs 18 satellites)
    contradicted = detector.detect_single(
        question_id="Q066",
        model_id="mock-b",
        generated_answer="Phase 2 added 24 synthetic-aperture radar satellites in 2023.",
        context=context,
        category="Context-based Question Answering",
        hallucination_test_type="context_supported_precision",
    )
    assert contradicted.claims[0].verification_label == LABEL_CONTRADICTED
    assert contradicted.faithfulness.contradicted_count == 1
    assert contradicted.faithfulness.contradiction_rate == pytest.approx(1.0)
    assert contradicted.faithfulness.faithfulness_ratio == pytest.approx(0.0)
    assert contradicted.hallucination_detected is True
    assert "context_contradiction" in contradicted.hallucination_types
    assert "fabricated_number_or_date" in contradicted.hallucination_types


def test_partially_supported_and_unsupported_claims_not_called_contradicted() -> None:
    """
    Verify unsupported claims receive NOT_ENOUGH_INFORMATION rather than CONTRADICTED,
    confidence does not hit 1.0 merely on zero retrieval, and broad context_hallucination_rate
    combines contradiction_rate + unsupported_claim_rate cleanly.
    """
    context = "Phase 2 added 18 synthetic-aperture radar satellites in 2023."
    detector = HallucinationDetector(embedder=MockTopicEmbedder())

    partial = detector.detect_single(
        question_id="Q066",
        model_id="mock-a",
        generated_answer=(
            "Phase 2 added 18 synthetic-aperture radar satellites in 2023. "
            "The total program budget was 450 million dollars."
        ),
        context=context,
    )
    assert partial.faithfulness.total_factual_claims == 2
    assert partial.faithfulness.supported_count == 1
    assert partial.faithfulness.unsupported_count == 1
    assert partial.faithfulness.contradicted_count == 0
    assert partial.faithfulness.faithfulness_ratio == pytest.approx(0.5)
    assert partial.faithfulness.contradiction_rate == pytest.approx(0.0)
    assert partial.faithfulness.unsupported_claim_rate == pytest.approx(0.5)
    assert partial.faithfulness.context_hallucination_rate == pytest.approx(0.5)
    assert partial.context_hallucination_rate == pytest.approx(
        partial.faithfulness.contradiction_rate + partial.faithfulness.unsupported_claim_rate
    )

    unsupported_claim = partial.claims[1]
    assert unsupported_claim.verification_label == LABEL_NOT_ENOUGH_INFO
    assert 0.50 <= unsupported_claim.confidence <= 0.85
    assert 0.0 <= unsupported_claim.evidence_strength < 0.30
    assert (
        unsupported_claim.details.get("confidence_interpretation")
        == "classification_confidence_insufficient_evidence_not_falsehood"
    )
    assert partial.hallucination_detected is True
    assert "unsupported_context_claim" in partial.hallucination_types
    assert "context_contradiction" not in partial.hallucination_types


def test_mixed_contradicted_and_unsupported_broad_hallucination_rate() -> None:
    """Verify contradiction_rate, unsupported_claim_rate, and context_hallucination_rate together."""
    context = "Phase 2 added 18 synthetic-aperture radar satellites in 2023."
    detector = HallucinationDetector(embedder=MockTopicEmbedder())

    mixed = detector.detect_single(
        question_id="Q066",
        model_id="mock-b",
        generated_answer=(
            "Phase 2 added 24 synthetic-aperture radar satellites in 2023. "
            "The program cost 450 million dollars."
        ),
        context=context,
    )
    assert mixed.faithfulness.total_factual_claims == 2
    assert mixed.faithfulness.supported_count == 0
    assert mixed.faithfulness.contradicted_count == 1
    assert mixed.faithfulness.unsupported_count == 1
    assert mixed.faithfulness.faithfulness_ratio == pytest.approx(0.0)
    assert mixed.faithfulness.contradiction_rate == pytest.approx(0.5)
    assert mixed.faithfulness.unsupported_claim_rate == pytest.approx(0.5)
    assert mixed.faithfulness.context_hallucination_rate == pytest.approx(1.0)
    assert mixed.context_hallucination_rate == pytest.approx(1.0)
    assert mixed.claims[0].verification_label == LABEL_CONTRADICTED
    assert mixed.claims[0].evidence_strength > 0.0
    assert mixed.claims[1].verification_label == LABEL_NOT_ENOUGH_INFO
    assert mixed.claims[1].confidence < 1.0


def test_counterfactual_context_evaluated_strictly_against_context() -> None:
    """Verify counterfactual context overrides real-world truth and is flagged cleanly."""
    cf_context = (
        "Simulation Rulebook Alpha: Within this simulated environment, pure water boils at "
        "72 degrees Celsius at sea level, and the capital of France is Lyon."
    )
    detector = HallucinationDetector(
        embedder=MockTopicEmbedder(),
        nli_scorer=MockNLIScorer(),
    )

    faithful_cf = detector.detect_single(
        question_id="Q079",
        model_id="mock-a",
        generated_answer="Pure water boils at 72 degrees Celsius at sea level and the capital of France is Lyon.",
        context=cf_context,
        category="Context-based Question Answering",
        hallucination_test_type="counterfactual_context",
        expected_behavior="follow_counterfactual_context",
    )
    assert faithful_cf.is_counterfactual_context is True
    assert faithful_cf.faithfulness.total_factual_claims == 2
    assert faithful_cf.faithfulness.supported_count == 2
    assert faithful_cf.hallucination_detected is False

    # Model ignoring counterfactual context and giving real-world 100 C / Paris
    unfaithful_cf = detector.detect_single(
        question_id="Q079",
        model_id="mock-b",
        generated_answer="Water boils at 100 degrees Celsius and the capital of France is Paris.",
        context=cf_context,
        category="Context-based Question Answering",
        hallucination_test_type="counterfactual_context",
        expected_behavior="follow_counterfactual_context",
    )
    assert unfaithful_cf.is_counterfactual_context is True
    assert unfaithful_cf.faithfulness.contradicted_count == 2
    assert unfaithful_cf.hallucination_detected is True


def test_irrelevant_context_empty_response_and_no_factual_claims() -> None:
    """Verify irrelevant context refusal vs fabrication, empty output, and filler-only output."""
    focaccia_ctx = (
        "Bakers in northern Italy prepare focaccia using flour, olive oil, sea salt, "
        "and rosemary, baking the dough at 230 degrees Celsius."
    )
    detector = HallucinationDetector(embedder=MockTopicEmbedder())

    # Appropriate refusal on irrelevant context -> no hallucination
    refusal_rec = detector.detect_single(
        question_id="Q084",
        model_id="mock-a",
        generated_answer="The provided context does not contain information about the melting point of titanium.",
        context=focaccia_ctx,
        hallucination_test_type="misleading_irrelevant_context",
        expected_behavior="acknowledge_insufficient_info",
    )
    assert refusal_rec.faithfulness.status == "appropriate_refusal"
    assert refusal_rec.faithfulness.faithfulness_ratio is None
    assert refusal_rec.hallucination_detected is False

    # Guessing titanium melting point without evidence -> NOT_ENOUGH_INFORMATION
    guess_rec = detector.detect_single(
        question_id="Q084",
        model_id="mock-b",
        generated_answer="Titanium melts at 1668 degrees Celsius.",
        context=focaccia_ctx,
        hallucination_test_type="misleading_irrelevant_context",
    )
    assert guess_rec.claims[0].verification_label == LABEL_NOT_ENOUGH_INFO
    assert guess_rec.hallucination_detected is True

    # Empty response -> no zero-division
    empty_rec = detector.detect_single(
        question_id="Q066",
        model_id="mock-a",
        generated_answer="   ",
        context=focaccia_ctx,
    )
    assert empty_rec.faithfulness.status == "empty_response"
    assert empty_rec.faithfulness.faithfulness_ratio is None

    # Conversational filler only -> no factual claims
    filler_rec = detector.detect_single(
        question_id="Q066",
        model_id="mock-a",
        generated_answer="Sure! Here is the answer. Hope this helps.",
        context=focaccia_ctx,
    )
    assert filler_rec.faithfulness.status == "no_factual_claims"
    assert filler_rec.faithfulness.faithfulness_ratio is None


def test_missing_context_skips_faithfulness_and_distinguishes_adversarial_traps() -> None:
    """Verify context-free questions skip context-faithfulness and flag adversarial traps separately."""
    detector = HallucinationDetector(embedder=MockTopicEmbedder())

    # Closed-book factual answer without context -> faithfulness skipped (None)
    closed_book = detector.detect_single(
        question_id="Q001",
        model_id="mock-a",
        generated_answer="Canberra is the capital city of Australia.",
        context="",
        reference_answer="Canberra is the capital city of Australia.",
        category="General Knowledge",
        hallucination_test_type="factual_precision",
        expected_behavior="factual_answer",
    )
    assert closed_book.has_context is False
    assert closed_book.faithfulness.status == "no_context"
    assert closed_book.faithfulness.faithfulness_ratio is None
    assert closed_book.faithfulness.contradiction_rate is None
    assert closed_book.faithfulness.unsupported_claim_rate is None
    assert closed_book.faithfulness.context_hallucination_rate is None
    assert closed_book.context_hallucination_rate is None
    assert closed_book.claims[0].confidence == 0.0
    assert closed_book.claims[0].evidence_strength == 0.0
    assert closed_book.hallucination_detected is False

    # Fabricated citation answer when refusal/uncertainty was expected
    fake_doi = detector.detect_single(
        question_id="Q098",
        model_id="mock-b",
        generated_answer="The DOI is 10.1038/s41586-019-9999-1 on pages 45-52.",
        context="",
        reference_answer="I cannot verify any 2019 Nature paper by Dr. Arthur Pendelton.",
        category="Adversarial and Hallucination Testing",
        hallucination_test_type="fabricated_citation_or_number",
        expected_behavior="acknowledge_uncertainty",
    )
    assert fake_doi.has_context is False
    assert fake_doi.faithfulness.faithfulness_ratio is None
    assert fake_doi.hallucination_detected is True
    assert "possible_fabricated_citation_or_fact" in fake_doi.hallucination_types


def test_batch_integration_and_json_csv_persistence(tmp_path) -> None:
    """Verify Phase 4 GenerationResult integration and JSON/CSV save/reload."""
    dataset_df = pd.DataFrame(
        [
            {
                "question_id": "Q066",
                "question": "How many satellites in Phase 2?",
                "context": "Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
                "reference_answer": "18 synthetic-aperture radar satellites.",
                "category": "Context-based Question Answering",
                "hallucination_test_type": "context_supported_precision",
                "expected_behavior": "context_grounded_answer",
            },
            {
                "question_id": "Q001",
                "question": "Capital of Australia?",
                "context": "",
                "reference_answer": "Canberra.",
                "category": "General Knowledge",
                "hallucination_test_type": "factual_precision",
                "expected_behavior": "factual_answer",
            },
        ]
    )

    gen_results = [
        GenerationResult(
            question_id="Q066",
            model_key="mock-a",
            model_id="org/mock-a",
            backend="huggingface_local",
            question="How many satellites in Phase 2?",
            context="Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
            prompt_used="...",
            generated_answer="Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
            generation_time_seconds=0.40,
            status="success",
        ),
        GenerationResult(
            question_id="Q001",
            model_key="mock-a",
            model_id="org/mock-a",
            backend="huggingface_local",
            question="Capital of Australia?",
            context="",
            prompt_used="...",
            generated_answer="Canberra is the capital city of Australia.",
            generation_time_seconds=0.20,
            status="success",
        ),
    ]

    detector = HallucinationDetector(embedder=MockTopicEmbedder())
    records = detector.detect_from_generation_results(gen_results, dataset_df)
    assert len(records) == 2
    assert records[0].faithfulness.faithfulness_ratio == pytest.approx(1.0)
    assert records[1].faithfulness.status == "no_context"

    paths = save_hallucination_results(records, output_dir=tmp_path, run_name="hallucination_test")
    assert paths["json"].exists() and paths["csv"].exists()

    reloaded = load_hallucination_results(paths["json"])
    assert len(reloaded) == 2
    assert isinstance(reloaded[0], HallucinationDetectionRecord)
    df_csv = pd.read_csv(paths["csv"])
    assert len(df_csv) == 2
    assert "faithfulness_ratio" in df_csv.columns
    assert "context_hallucination_rate" in df_csv.columns
