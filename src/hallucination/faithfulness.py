"""
Claim-level faithfulness metrics calculation.

Computes transparent counts and normalized rates strictly for context-bearing
questions that contain extracted factual claims, avoiding zero-division and
never calculating context faithfulness when no context is supplied.

Preserves separately:
- faithfulness_ratio: supported_count / total_factual_claims
- contradiction_rate: contradicted_count / total_factual_claims
- unsupported_claim_rate: unsupported_count / total_factual_claims
- context_hallucination_rate: (contradicted_count + unsupported_count) / total_factual_claims
  (a broad unsupported-by-context measure; does NOT establish that every unsupported
  claim is factually false).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.hallucination.claim_verifier import (
    LABEL_CONTRADICTED,
    LABEL_NOT_ENOUGH_INFO,
    LABEL_SUPPORTED,
    ClaimVerificationResult,
)


@dataclass(frozen=True)
class FaithfulnessMetrics:
    """
    Individual claim-level faithfulness counts and normalized ratios.

    Note: `context_hallucination_rate` combines contradicted and unsupported claims
    as a broad unsupported-by-context measure and does not establish that unsupported
    claims are factually false.
    """

    question_id: str
    model_id: str
    has_context: bool
    supported_count: int
    contradicted_count: int
    unsupported_count: int
    total_factual_claims: int
    faithfulness_ratio: Optional[float]
    contradiction_rate: Optional[float]
    unsupported_claim_rate: Optional[float]
    context_hallucination_rate: Optional[float]
    status: str  # "computed", "no_context", "appropriate_refusal", "no_factual_claims", "empty_response"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize faithfulness metrics to a dictionary."""
        return asdict(self)


def compute_faithfulness_metrics(
    question_id: str,
    model_id: str,
    has_context: bool,
    verification_results: Sequence[ClaimVerificationResult],
    extraction_status: str = "extracted",
) -> FaithfulnessMetrics:
    """
    Compute claim-level faithfulness counts and rates without silent aggregation.

    Returns None ratios when no context was supplied or when no factual claims exist.
    """
    if not has_context:
        return FaithfulnessMetrics(
            question_id=question_id,
            model_id=model_id,
            has_context=False,
            supported_count=0,
            contradicted_count=0,
            unsupported_count=0,
            total_factual_claims=len(verification_results),
            faithfulness_ratio=None,
            contradiction_rate=None,
            unsupported_claim_rate=None,
            context_hallucination_rate=None,
            status="no_context",
            notes=["Context faithfulness is not calculated for questions without supplied context."],
        )

    if extraction_status == "empty_response":
        return FaithfulnessMetrics(
            question_id=question_id,
            model_id=model_id,
            has_context=True,
            supported_count=0,
            contradicted_count=0,
            unsupported_count=0,
            total_factual_claims=0,
            faithfulness_ratio=None,
            contradiction_rate=None,
            unsupported_claim_rate=None,
            context_hallucination_rate=None,
            status="empty_response",
            notes=["Response was empty; no factual claims to verify."],
        )

    total = len(verification_results)
    if total == 0:
        if extraction_status == "refusal_only":
            return FaithfulnessMetrics(
                question_id=question_id,
                model_id=model_id,
                has_context=True,
                supported_count=0,
                contradicted_count=0,
                unsupported_count=0,
                total_factual_claims=0,
                faithfulness_ratio=None,
                contradiction_rate=None,
                unsupported_claim_rate=None,
                context_hallucination_rate=None,
                status="appropriate_refusal",
                notes=["Response expressed insufficiency or refusal without ungrounded claims."],
            )
        return FaithfulnessMetrics(
            question_id=question_id,
            model_id=model_id,
            has_context=True,
            supported_count=0,
            contradicted_count=0,
            unsupported_count=0,
            total_factual_claims=0,
            faithfulness_ratio=None,
            contradiction_rate=None,
            unsupported_claim_rate=None,
            context_hallucination_rate=None,
            status="no_factual_claims",
            notes=["No verifiable factual claims were extracted from the response."],
        )

    supported = sum(1 for r in verification_results if r.verification_label == LABEL_SUPPORTED)
    contradicted = sum(1 for r in verification_results if r.verification_label == LABEL_CONTRADICTED)
    unsupported = sum(1 for r in verification_results if r.verification_label == LABEL_NOT_ENOUGH_INFO)

    notes: List[str] = []
    if unsupported > 0:
        notes.append(
            "context_hallucination_rate combines contradicted and unsupported claims as a broad "
            "unsupported-by-context measure; unsupported claims are unverified by context, not proven false."
        )

    return FaithfulnessMetrics(
        question_id=question_id,
        model_id=model_id,
        has_context=True,
        supported_count=supported,
        contradicted_count=contradicted,
        unsupported_count=unsupported,
        total_factual_claims=total,
        faithfulness_ratio=round(supported / total, 6),
        contradiction_rate=round(contradicted / total, 6),
        unsupported_claim_rate=round(unsupported / total, 6),
        context_hallucination_rate=round((contradicted + unsupported) / total, 6),
        status="computed",
        notes=notes,
    )
