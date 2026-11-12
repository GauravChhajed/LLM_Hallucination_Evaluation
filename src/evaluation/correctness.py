"""
Reference-based correctness assessment layer.

Combines lexical overlap, key information coverage, numeric/entity consistency,
negation/refusal alignment, and semantic similarity into transparent diagnostic
signals plus an optional, configurable rule-based decision framework.

Note: Neither lexical overlap nor embedding cosine similarity alone proves
factual correctness.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.evaluation.lexical_metrics import LexicalMetricsResult, tokenize_text

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
        "from", "has", "have", "had", "he", "her", "hers", "him", "his", "in",
        "into", "is", "it", "its", "of", "on", "or", "our", "she", "that", "the",
        "their", "them", "there", "these", "they", "this", "those", "to", "was",
        "were", "which", "who", "will", "with", "would", "according", "passage",
        "provided", "context", "question", "answer", "based", "strictly",
    }
)

_NUMBER_PATTERN = re.compile(r"(?<![a-zA-Z])-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?%?")

_REFUSAL_CUES: tuple[str, ...] = (
    "cannot be determined",
    "cannot verify",
    "cannot confirm",
    "cannot answer",
    "cannot conclude",
    "does not state",
    "does not contain",
    "does not mention",
    "does not specify",
    "does not provide",
    "not enough information",
    "insufficient information",
    "no information",
    "not possible to know",
    "false premise",
    "ambiguous",
    "uncertain",
    "unknown",
    "clarify",
    "clarification",
    "not exist",
    "non-existent",
    "never joined",
    "did not win",
    "not a chemical element",
    "zero natural moons",
    "no natural moons",
)


@dataclass(frozen=True)
class CorrectnessConfig:
    """Configurable weights and decision thresholds for reference correctness assessment."""

    semantic_weight: float = 0.40
    key_info_weight: float = 0.40
    token_f1_weight: float = 0.20
    correct_threshold: float = 0.72
    partial_threshold: float = 0.42
    require_numeric_consistency: bool = True
    require_refusal_alignment: bool = True


@dataclass(frozen=True)
class CorrectnessAssessment:
    """Transparent multi-signal assessment of reference-based correctness."""

    correctness_score: Optional[float]
    correctness_label: str
    key_info_coverage: float
    numeric_match_ratio: Optional[float]
    has_numeric_contradiction: bool
    refusal_expected: bool
    refusal_detected: bool
    refusal_mismatch: bool
    matched_key_terms: List[str] = field(default_factory=list)
    missing_key_terms: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize assessment to a plain dictionary."""
        return asdict(self)


def extract_numeric_tokens(text: str) -> Set[str]:
    """
    Extract normalized numeric strings (integers, decimals, fractions, percentages).

    Strips thousands commas and trailing '%' so '1,000' -> '1000'.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return set()
    matches = _NUMBER_PATTERN.findall(cleaned)
    normalized: Set[str] = set()
    for match in matches:
        token = match.replace(",", "").rstrip("%")
        # Normalize integer-equivalent decimals such as 7.0 -> 7
        if re.fullmatch(r"-?\d+\.0+", token):
            token = token.split(".")[0]
        normalized.add(token)
    return normalized


def extract_key_terms(text: str) -> List[str]:
    """Extract ordered unique content terms (retaining numbers and negation cues)."""
    tokens = tokenize_text(text, remove_articles=True)
    seen: Set[str] = set()
    key_terms: List[str] = []
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        if len(tok) <= 1 and not tok.isdigit():
            continue
        if tok not in seen:
            seen.add(tok)
            key_terms.append(tok)
    return key_terms


def detects_refusal_or_correction(text: str) -> bool:
    """Check whether text contains explicit uncertainty, refusal, or premise-rejection cues."""
    lower = (text or "").strip().lower()
    if not lower:
        return False
    return any(cue in lower for cue in _REFUSAL_CUES)


def is_refusal_or_correction_expected(
    reference_answer: str,
    expected_behavior: Optional[str] = None,
    hallucination_test_type: Optional[str] = None,
) -> bool:
    """Determine if the question expects refusal, uncertainty, clarification, or premise rejection."""
    refusal_behaviors = {
        "reject_false_premise",
        "acknowledge_insufficient_info",
        "acknowledge_uncertainty",
        "clarify_ambiguity",
    }
    refusal_types = {
        "false_premise",
        "insufficient_information",
        "ambiguous_question",
        "unanswerable_from_context",
        "uncertainty_acknowledgment",
        "fabricated_citation_or_number",
    }
    if expected_behavior and expected_behavior in refusal_behaviors:
        return True
    if hallucination_test_type and hallucination_test_type in refusal_types:
        return True
    return detects_refusal_or_correction(reference_answer)


class ReferenceCorrectnessEvaluator:
    """
    Combines lexical overlap, key-term coverage, numeric verification, refusal
    alignment, and semantic similarity without equating similarity with truth.
    """

    def __init__(self, config: Optional[CorrectnessConfig] = None) -> None:
        self.config = config or CorrectnessConfig()

    def assess(
        self,
        generated_answer: str,
        reference_answer: str,
        lexical: LexicalMetricsResult,
        semantic_similarity: Optional[float],
        question_type: Optional[str] = None,
        expected_behavior: Optional[str] = None,
        hallucination_test_type: Optional[str] = None,
    ) -> CorrectnessAssessment:
        """Produce a transparent correctness assessment from individual signals."""
        clean_gen = (generated_answer or "").strip()
        clean_ref = (reference_answer or "").strip()

        if not clean_gen or not clean_ref:
            return CorrectnessAssessment(
                correctness_score=None,
                correctness_label="unassessed",
                key_info_coverage=0.0,
                numeric_match_ratio=None,
                has_numeric_contradiction=False,
                refusal_expected=False,
                refusal_detected=False,
                refusal_mismatch=False,
                notes=["Missing generated or reference answer; correctness cannot be assessed."],
            )

        # 1. Key-information coverage
        ref_key_terms = extract_key_terms(clean_ref)
        gen_token_set = set(tokenize_text(clean_gen, remove_articles=True))
        matched_terms = [t for t in ref_key_terms if t in gen_token_set]
        missing_terms = [t for t in ref_key_terms if t not in gen_token_set]
        key_info_coverage = (
            round(len(matched_terms) / len(ref_key_terms), 6) if ref_key_terms else 0.0
        )

        # 2. Numeric consistency check (critical for math, dates, quantities)
        ref_nums = extract_numeric_tokens(clean_ref)
        gen_nums = extract_numeric_tokens(clean_gen)
        numeric_match_ratio: Optional[float] = None
        has_numeric_contradiction = False
        notes: List[str] = []

        if ref_nums:
            matched_nums = ref_nums & gen_nums
            numeric_match_ratio = round(len(matched_nums) / len(ref_nums), 6)
            # Flag contradiction if expected numbers are missing or conflicting numbers appear
            if len(matched_nums) < len(ref_nums) and (
                len(gen_nums - ref_nums) > 0 or question_type == "Mathematics"
            ):
                has_numeric_contradiction = True
                notes.append(
                    f"Numeric mismatch detected: expected {sorted(ref_nums)}, "
                    f"found {sorted(gen_nums)}."
                )

        # 3. Refusal / uncertainty / false-premise alignment check
        refusal_expected = is_refusal_or_correction_expected(
            clean_ref,
            expected_behavior=expected_behavior,
            hallucination_test_type=hallucination_test_type,
        )
        refusal_detected = detects_refusal_or_correction(clean_gen)
        refusal_mismatch = refusal_expected and not refusal_detected
        if refusal_mismatch:
            notes.append(
                "Expected uncertainty/insufficiency/premise-rejection behavior was not detected."
            )

        # 4. Exact match shortcut
        if lexical.normalized_exact_match == 1.0:
            notes.append("Normalized exact match with reference answer.")
            return CorrectnessAssessment(
                correctness_score=1.0,
                correctness_label="correct",
                key_info_coverage=1.0,
                numeric_match_ratio=1.0 if ref_nums else None,
                has_numeric_contradiction=False,
                refusal_expected=refusal_expected,
                refusal_detected=refusal_detected,
                refusal_mismatch=False,
                matched_key_terms=matched_terms,
                missing_key_terms=missing_terms,
                notes=notes,
            )

        # 5. Transparent composite score (re-normalized if semantic_similarity is unavailable)
        if semantic_similarity is not None:
            norm_sem = max(0.0, min(1.0, float(semantic_similarity)))
            raw_score = (
                self.config.semantic_weight * norm_sem
                + self.config.key_info_weight * key_info_coverage
                + self.config.token_f1_weight * lexical.token_f1
            )
        else:
            denom = self.config.key_info_weight + self.config.token_f1_weight
            raw_score = (
                (self.config.key_info_weight * key_info_coverage)
                + (self.config.token_f1_weight * lexical.token_f1)
            ) / denom
            notes.append("Semantic similarity unavailable; score computed from lexical & key-term metrics.")

        # Apply documented penalties when factual/refusal contradictions are observed
        if self.config.require_numeric_consistency and has_numeric_contradiction:
            raw_score = min(raw_score, 0.30)
            notes.append(
                "High lexical/semantic overlap capped because numeric values contradict the reference."
            )

        if self.config.require_refusal_alignment and refusal_mismatch:
            raw_score = min(raw_score, 0.35)

        score = round(max(0.0, min(1.0, raw_score)), 6)

        if has_numeric_contradiction and self.config.require_numeric_consistency:
            label = "contradictory_numeric"
        elif score >= self.config.correct_threshold:
            label = "correct"
        elif score >= self.config.partial_threshold:
            label = "partially_correct"
        else:
            label = "incorrect"

        return CorrectnessAssessment(
            correctness_score=score,
            correctness_label=label,
            key_info_coverage=key_info_coverage,
            numeric_match_ratio=numeric_match_ratio,
            has_numeric_contradiction=has_numeric_contradiction,
            refusal_expected=refusal_expected,
            refusal_detected=refusal_detected,
            refusal_mismatch=refusal_mismatch,
            matched_key_terms=matched_terms,
            missing_key_terms=missing_terms,
            notes=notes,
        )
