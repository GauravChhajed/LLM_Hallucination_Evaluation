"""Claim-level faithfulness and hallucination detection package."""

from src.hallucination.claim_extractor import (
    ClaimExtractionResult,
    ClaimExtractor,
    ExtractedClaim,
    classify_claim_type,
    is_conversational_filler,
    is_refusal_or_uncertainty_statement,
    split_sentences,
)
from src.hallucination.claim_verifier import (
    LABEL_CONTRADICTED,
    LABEL_NOT_ENOUGH_INFO,
    LABEL_SUPPORTED,
    BaseNLIScorer,
    ClaimVerificationConfig,
    ClaimVerificationResult,
    ClaimVerifier,
    CrossEncoderNLIScorer,
    NLIScores,
)
from src.hallucination.evidence_retriever import (
    EvidenceRetrievalResult,
    EvidenceRetriever,
    RetrievedEvidence,
)
from src.hallucination.faithfulness import FaithfulnessMetrics, compute_faithfulness_metrics
from src.hallucination.hallucination_detector import (
    HallucinationDetectionRecord,
    HallucinationDetector,
    hallucination_records_to_dataframe,
    load_hallucination_results,
    save_hallucination_results,
)

__all__ = [
    "BaseNLIScorer",
    "ClaimExtractionResult",
    "ClaimExtractor",
    "ClaimVerificationConfig",
    "ClaimVerificationResult",
    "ClaimVerifier",
    "CrossEncoderNLIScorer",
    "EvidenceRetrievalResult",
    "EvidenceRetriever",
    "ExtractedClaim",
    "FaithfulnessMetrics",
    "HallucinationDetectionRecord",
    "HallucinationDetector",
    "LABEL_CONTRADICTED",
    "LABEL_NOT_ENOUGH_INFO",
    "LABEL_SUPPORTED",
    "NLIScores",
    "RetrievedEvidence",
    "classify_claim_type",
    "compute_faithfulness_metrics",
    "hallucination_records_to_dataframe",
    "is_conversational_filler",
    "is_refusal_or_uncertainty_statement",
    "load_hallucination_results",
    "save_hallucination_results",
    "split_sentences",
]
