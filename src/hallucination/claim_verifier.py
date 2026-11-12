"""
Claim-level verification against retrieved context evidence.

Labels each extracted factual claim as SUPPORTED, CONTRADICTED, or
NOT_ENOUGH_INFORMATION while recording retrieved evidence, confidence score,
and human-readable justification. Supports optional NLI cross-encoder scoring
as well as mocked scorers and deterministic numeric/negation consistency guards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.evaluation.correctness import extract_key_terms, extract_numeric_tokens
from src.evaluation.lexical_metrics import tokenize_text
from src.hallucination.claim_extractor import ExtractedClaim
from src.hallucination.evidence_retriever import EvidenceRetrievalResult
from src.utils.config import AppConfig, get_config, resolve_device

LABEL_SUPPORTED = "SUPPORTED"
LABEL_CONTRADICTED = "CONTRADICTED"
LABEL_NOT_ENOUGH_INFO = "NOT_ENOUGH_INFORMATION"

_NEGATION_TOKENS: frozenset[str] = frozenset(
    {"no", "not", "never", "none", "zero", "neither", "nor", "cannot", "cant", "without"}
)

_GENERIC_UNIT_WORDS: frozenset[str] = frozenset(
    {
        "degree", "degrees", "celsius", "fahrenheit", "kelvin", "percent", "percentage",
        "million", "billion", "thousand", "hundred", "dollar", "dollars", "usd",
        "km", "cm", "mm", "kg", "gram", "grams", "tonnes", "hour", "hours",
        "minute", "minutes", "second", "seconds", "year", "years", "day", "days",
        "month", "months", "watt", "watts", "hz", "gb", "mb",
    }
)


@dataclass(frozen=True)
class NLIScores:
    """Normalized NLI probability distribution over entailment, contradiction, and neutral."""

    entailment: float
    contradiction: float
    neutral: float


@dataclass(frozen=True)
class ClaimVerificationConfig:
    """Documented thresholds for NLI and hybrid claim verification."""

    entailment_threshold: float = 0.55
    contradiction_threshold: float = 0.60
    embedding_support_threshold: float = 0.55
    key_term_support_threshold: float = 0.50


@dataclass(frozen=True)
class ClaimVerificationResult:
    """
    Verification outcome for a single extracted factual claim.

    Attributes:
        confidence: Classification confidence in `verification_label` in [0.0, 1.0].
            For NOT_ENOUGH_INFORMATION, this reflects confidence that the context
            lacks sufficient evidence—it NEVER implies that the claim is false.
        evidence_strength: Observed strength of supporting/retrieved context evidence
            in [0.0, 1.0] (0.0 when no context or no relevant evidence was retrieved).
    """

    claim_id: str
    question_id: str
    model_id: str
    claim_text: str
    claim_type: str
    retrieved_evidence: List[str]
    evidence_scores: List[float]
    verification_label: str  # SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFORMATION
    confidence: float
    explanation: str
    verifier_backend: str
    evidence_strength: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize verification result to a plain dictionary."""
        return asdict(self)


class BaseNLIScorer(ABC):
    """Abstract NLI scorer taking (premise, hypothesis) pairs."""

    model_name: str

    @abstractmethod
    def score(self, premise: str, hypothesis: str) -> NLIScores:
        """Return entailment, contradiction, and neutral probabilities summing to ~1.0."""


class CrossEncoderNLIScorer(BaseNLIScorer):
    """
    Lazy-loading NLI cross-encoder scorer (default: cross-encoder/nli-deberta-v3-small).

    Does not download weights until `score()` is explicitly invoked.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        preferred_device: str = "auto",
        app_config: Optional[AppConfig] = None,
        local_files_only: bool = False,
        model_loader: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.app_config = app_config or get_config()
        self.model_name = model_name or self.app_config.evaluation.nli_model_id
        self.preferred_device = preferred_device
        self.local_files_only = local_files_only
        self._model_loader = model_loader
        self._model: Any = None
        self._tokenizer: Any = None

    def ensure_loaded(self) -> None:
        """Lazily load NLI tokenizer and sequence classification model."""
        if self._model is not None and self._tokenizer is not None:
            return

        device = resolve_device(self.preferred_device)
        cache_dir = str(self.app_config.paths.hf_cache_dir)
        if self._model_loader is not None:
            self._tokenizer, self._model = self._model_loader(
                self.model_name,
                device=device,
                cache_dir=cache_dir,
                local_files_only=self.local_files_only,
            )
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        ).to(device)
        self._model.eval()

    def score(self, premise: str, hypothesis: str) -> NLIScores:
        """Compute softmax NLI probabilities for a premise-hypothesis pair."""
        import torch

        self.ensure_loaded()
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        device = next(self._model.parameters()).device if hasattr(self._model, "parameters") else "cpu"
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)

        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        id2label = getattr(getattr(self._model, "config", None), "id2label", None) or {
            0: "contradiction",
            1: "entailment",
            2: "neutral",
        }
        mapped: Dict[str, float] = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for idx, p in enumerate(probs):
            label_name = str(id2label.get(idx, "")).lower()
            if "contradict" in label_name:
                mapped["contradiction"] = float(p)
            elif "entail" in label_name:
                mapped["entailment"] = float(p)
            else:
                mapped["neutral"] = float(p)

        return NLIScores(
            entailment=round(mapped["entailment"], 6),
            contradiction=round(mapped["contradiction"], 6),
            neutral=round(mapped["neutral"], 6),
        )


def _has_negation_flip(claim_text: str, evidence_text: str) -> bool:
    """Detect opposing negation polarity when claim and evidence share core terms."""
    claim_tokens = set(tokenize_text(claim_text, remove_articles=True))
    ev_tokens = set(tokenize_text(evidence_text, remove_articles=True))
    claim_neg = bool(claim_tokens & _NEGATION_TOKENS)
    ev_neg = bool(ev_tokens & _NEGATION_TOKENS)
    if claim_neg == ev_neg:
        return False

    claim_content = set(extract_key_terms(claim_text)) - _NEGATION_TOKENS
    ev_content = set(extract_key_terms(evidence_text)) - _NEGATION_TOKENS
    if not claim_content:
        return False
    overlap_ratio = len(claim_content & ev_content) / len(claim_content)
    return overlap_ratio >= 0.45


def _detect_numeric_conflict(claim_text: str, evidence_text: str) -> Optional[str]:
    """
    Detect numerical/year contradictions between a claim and closely matching evidence.

    Returns a descriptive conflict message if conflicting numbers are detected, else None.
    """
    claim_nums = extract_numeric_tokens(claim_text)
    ev_nums = extract_numeric_tokens(evidence_text)
    if not claim_nums or not ev_nums:
        return None

    missing_in_ev = claim_nums - ev_nums
    if not missing_in_ev:
        return None

    # Require substantive shared non-numeric, non-unit subject overlap before calling it
    # a contradiction rather than an unrelated measurement on a different subject.
    claim_words = {
        t for t in extract_key_terms(claim_text)
        if t not in claim_nums and t not in _GENERIC_UNIT_WORDS
    }
    ev_words = {
        t for t in extract_key_terms(evidence_text)
        if t not in ev_nums and t not in _GENERIC_UNIT_WORDS
    }
    if not claim_words:
        return None
    shared_subject = claim_words & ev_words
    topic_overlap = len(shared_subject) / len(claim_words)

    if len(shared_subject) >= 1 and topic_overlap >= 0.40:
        return (
            f"Claim numeric/date value(s) {sorted(missing_in_ev)} conflict with "
            f"context evidence value(s) {sorted(ev_nums)}."
        )
    return None


class ClaimVerifier:
    """
    Verifies atomic claims against retrieved context evidence.

    Combines optional NLI scoring with deterministic numeric/negation checks
    and embedding-evidence alignment. Never equates low similarity with contradiction.
    """

    def __init__(
        self,
        nli_scorer: Optional[BaseNLIScorer] = None,
        config: Optional[ClaimVerificationConfig] = None,
    ) -> None:
        self.nli_scorer = nli_scorer
        self.config = config or ClaimVerificationConfig()

    def verify(
        self,
        claim: ExtractedClaim,
        retrieval: EvidenceRetrievalResult,
    ) -> ClaimVerificationResult:
        """Verify a single claim using its retrieved context evidence."""
        if not retrieval.has_context:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                question_id=claim.question_id,
                model_id=claim.model_id,
                claim_text=claim.claim_text,
                claim_type=claim.claim_type,
                retrieved_evidence=[],
                evidence_scores=[],
                verification_label=LABEL_NOT_ENOUGH_INFO,
                confidence=0.0,
                evidence_strength=0.0,
                explanation="No context was supplied for this question; claim cannot be verified against context.",
                verifier_backend=self._backend_name(),
                details={"confidence_interpretation": "not_applicable_no_context"},
            )

        evidence_items = retrieval.retrieved_evidence
        if not evidence_items:
            best_sim_clamped = round(max(0.0, float(retrieval.best_similarity)), 6)
            # Bounded in [0.50, 0.85]: reflects confidence that context lacks evidence,
            # never 1.0 merely from zero similarity, and never implies falsehood.
            nei_conf = round(max(0.50, min(0.85, 0.85 - best_sim_clamped)), 6)
            best_note = (
                f" Highest context sentence similarity was {retrieval.best_similarity:.2f}."
                if retrieval.best_evidence is not None
                else ""
            )
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                question_id=claim.question_id,
                model_id=claim.model_id,
                claim_text=claim.claim_text,
                claim_type=claim.claim_type,
                retrieved_evidence=[],
                evidence_scores=[],
                verification_label=LABEL_NOT_ENOUGH_INFO,
                confidence=nei_conf,
                evidence_strength=best_sim_clamped,
                explanation=(
                    "No context sentence exceeded the evidence retrieval threshold; "
                    "lack of supporting evidence is labeled NOT_ENOUGH_INFORMATION "
                    "(unverified against context, not established as false)."
                    f"{best_note}"
                ),
                verifier_backend=self._backend_name(),
                details={
                    "confidence_interpretation": "classification_confidence_insufficient_evidence_not_falsehood",
                    "best_context_similarity": best_sim_clamped,
                },
            )

        evidence_texts = [ev.sentence for ev in evidence_items]
        evidence_scores = [ev.similarity_score for ev in evidence_items]
        best_sim = max(0.0, max(evidence_scores)) if evidence_scores else 0.0
        combined_premise = " ".join(evidence_texts)

        # 1. Deterministic numeric/date conflict guard
        for ev_sent in evidence_texts:
            num_conflict = _detect_numeric_conflict(claim.claim_text, ev_sent)
            if num_conflict:
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    question_id=claim.question_id,
                    model_id=claim.model_id,
                    claim_text=claim.claim_text,
                    claim_type=claim.claim_type,
                    retrieved_evidence=evidence_texts,
                    evidence_scores=evidence_scores,
                    verification_label=LABEL_CONTRADICTED,
                    confidence=0.92,
                    evidence_strength=round(best_sim, 6),
                    explanation=num_conflict,
                    verifier_backend=self._backend_name(),
                    details={
                        "rule_trigger": "numeric_conflict",
                        "confidence_interpretation": "classification_confidence_contradiction",
                    },
                )

        # 2. Deterministic negation polarity flip guard
        for ev_sent in evidence_texts:
            if _has_negation_flip(claim.claim_text, ev_sent):
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    question_id=claim.question_id,
                    model_id=claim.model_id,
                    claim_text=claim.claim_text,
                    claim_type=claim.claim_type,
                    retrieved_evidence=evidence_texts,
                    evidence_scores=evidence_scores,
                    verification_label=LABEL_CONTRADICTED,
                    confidence=0.85,
                    evidence_strength=round(best_sim, 6),
                    explanation=(
                        "Negation polarity in the claim opposes the retrieved context evidence."
                    ),
                    verifier_backend=self._backend_name(),
                    details={
                        "rule_trigger": "negation_flip",
                        "confidence_interpretation": "classification_confidence_contradiction",
                    },
                )

        # 3. Optional NLI cross-encoder verification
        if self.nli_scorer is not None:
            try:
                nli = self.nli_scorer.score(combined_premise, claim.claim_text)
                if nli.contradiction >= self.config.contradiction_threshold:
                    return ClaimVerificationResult(
                        claim_id=claim.claim_id,
                        question_id=claim.question_id,
                        model_id=claim.model_id,
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        retrieved_evidence=evidence_texts,
                        evidence_scores=evidence_scores,
                        verification_label=LABEL_CONTRADICTED,
                        confidence=nli.contradiction,
                        evidence_strength=round(best_sim, 6),
                        explanation=(
                            f"NLI contradiction score ({nli.contradiction:.2f}) >= "
                            f"{self.config.contradiction_threshold:.2f}."
                        ),
                        verifier_backend=self._backend_name(),
                        details={
                            "nli_scores": asdict(nli),
                            "confidence_interpretation": "classification_confidence_contradiction",
                        },
                    )
                if nli.entailment >= self.config.entailment_threshold:
                    return ClaimVerificationResult(
                        claim_id=claim.claim_id,
                        question_id=claim.question_id,
                        model_id=claim.model_id,
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        retrieved_evidence=evidence_texts,
                        evidence_scores=evidence_scores,
                        verification_label=LABEL_SUPPORTED,
                        confidence=nli.entailment,
                        evidence_strength=round(max(best_sim, nli.entailment), 6),
                        explanation=(
                            f"NLI entailment score ({nli.entailment:.2f}) >= "
                            f"{self.config.entailment_threshold:.2f}."
                        ),
                        verifier_backend=self._backend_name(),
                        details={
                            "nli_scores": asdict(nli),
                            "confidence_interpretation": "classification_confidence_support",
                        },
                    )
                nei_nli_conf = round(
                    max(nli.neutral, 1.0 - max(nli.entailment, nli.contradiction)),
                    6,
                )
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    question_id=claim.question_id,
                    model_id=claim.model_id,
                    claim_text=claim.claim_text,
                    claim_type=claim.claim_type,
                    retrieved_evidence=evidence_texts,
                    evidence_scores=evidence_scores,
                    verification_label=LABEL_NOT_ENOUGH_INFO,
                    confidence=nei_nli_conf,
                    evidence_strength=round(min(best_sim, nli.entailment) if best_sim > 0 else nli.entailment, 6),
                    explanation=(
                        f"Neither NLI entailment ({nli.entailment:.2f}) nor contradiction "
                        f"({nli.contradiction:.2f}) met verification thresholds; claim is "
                        "unverified by context, not established as false."
                    ),
                    verifier_backend=self._backend_name(),
                    details={
                        "nli_scores": asdict(nli),
                        "confidence_interpretation": "classification_confidence_insufficient_evidence_not_falsehood",
                    },
                )
            except Exception as exc:
                # Fall through cleanly while noting NLI failure
                nli_error_note = f" (NLI fallback due to error: {exc})"
            else:
                nli_error_note = ""
        else:
            nli_error_note = ""

        # 4. Semantic-evidence + key-term consistency check (when NLI is not active)
        claim_terms = set(extract_key_terms(claim.claim_text))
        evidence_terms = set(extract_key_terms(combined_premise))
        term_coverage = (
            len(claim_terms & evidence_terms) / len(claim_terms) if claim_terms else 0.0
        )

        # Ensure any numbers appearing in the claim are also grounded in retrieved evidence
        claim_nums = extract_numeric_tokens(claim.claim_text)
        ev_nums = extract_numeric_tokens(combined_premise)
        numbers_grounded = claim_nums.issubset(ev_nums)

        best_sim = max(0.0, max(evidence_scores)) if evidence_scores else 0.0
        ev_strength = round(0.5 * best_sim + 0.5 * term_coverage, 6)
        if (
            best_sim >= self.config.embedding_support_threshold
            and term_coverage >= self.config.key_term_support_threshold
            and numbers_grounded
        ):
            confidence = round(min(1.0, ev_strength), 6)
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                question_id=claim.question_id,
                model_id=claim.model_id,
                claim_text=claim.claim_text,
                claim_type=claim.claim_type,
                retrieved_evidence=evidence_texts,
                evidence_scores=evidence_scores,
                verification_label=LABEL_SUPPORTED,
                confidence=confidence,
                evidence_strength=ev_strength,
                explanation=(
                    f"Supported by retrieved evidence (similarity={best_sim:.2f}, "
                    f"key-term coverage={term_coverage:.2f}).{nli_error_note}"
                ),
                verifier_backend=self._backend_name(),
                details={
                    "best_similarity": best_sim,
                    "key_term_coverage": round(term_coverage, 6),
                    "numbers_grounded": numbers_grounded,
                    "confidence_interpretation": "classification_confidence_support",
                },
            )

        nei_fallback_conf = round(max(0.50, min(0.85, 0.85 - 0.35 * term_coverage)), 6)
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            question_id=claim.question_id,
            model_id=claim.model_id,
            claim_text=claim.claim_text,
            claim_type=claim.claim_type,
            retrieved_evidence=evidence_texts,
            evidence_scores=evidence_scores,
            verification_label=LABEL_NOT_ENOUGH_INFO,
            confidence=nei_fallback_conf,
            evidence_strength=ev_strength,
            explanation=(
                f"Retrieved evidence did not sufficiently establish all claim details "
                f"(similarity={best_sim:.2f}, key-term coverage={term_coverage:.2f}, "
                f"numbers_grounded={numbers_grounded}); claim is unverified by context, "
                f"not established as false.{nli_error_note}"
            ),
            verifier_backend=self._backend_name(),
            details={
                "best_similarity": best_sim,
                "key_term_coverage": round(term_coverage, 6),
                "numbers_grounded": numbers_grounded,
                "confidence_interpretation": "classification_confidence_insufficient_evidence_not_falsehood",
            },
        )

    def _backend_name(self) -> str:
        if self.nli_scorer is not None:
            return f"hybrid+{self.nli_scorer.model_name}"
        return "hybrid_semantic_rule"
