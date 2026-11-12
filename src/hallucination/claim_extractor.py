"""
Atomic factual claim decomposition for generated LLM responses.

Splits responses into sentence- and clause-level candidate statements, filters
conversational filler and appropriate refusals/uncertainty acknowledgments, and
tags each extracted factual claim by type (numeric, temporal, named entity,
or general factual assertion).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

_REFUSAL_PATTERNS: tuple[str, ...] = (
    "cannot be determined",
    "cannot verify",
    "cannot confirm",
    "cannot answer",
    "cannot conclude",
    "cannot provide",
    "does not state",
    "does not contain",
    "does not mention",
    "does not specify",
    "does not provide",
    "do not have enough information",
    "not enough information",
    "insufficient information",
    "no information",
    "not possible to know",
    "unable to verify",
    "unable to determine",
    "false premise",
    "contains a false premise",
    "is ambiguous",
    "please clarify",
    "clarification is needed",
    "i am not sure",
    "i'm not sure",
    "i do not know",
    "i don't know",
    "unknown",
)

_CONVERSATIONAL_FILLER: frozenset[str] = frozenset(
    {
        "sure",
        "certainly",
        "of course",
        "here is the answer",
        "here is your answer",
        "thank you",
        "thanks",
        "hope this helps",
        "i hope this helps",
        "let me know if you need anything else",
    }
)

# Abbreviations where a period does not end a sentence
_ABBREV_PLACEHOLDER = "<PRD>"
_ABBREV_PATTERN = re.compile(
    r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|Fig|No|Vol|pp|et al)\.",
    re.IGNORECASE,
)

# Heuristic verb/copula detector to avoid splitting noun-phrase lists like "A and B"
_CLAUSE_VERB_PATTERN = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|do|does|did|can|could|will|would|"
    r"may|might|must|shall|should|boils|melts|freezes|equals|contains|includes|employs|"
    r"houses|won|lost|launched|opened|ended|fell|built|invented|painted|flows|separates|"
    r"catalyzes|carries|measures|costs|reached|reported|added|halted|replaced|relied)\b|"
    r"\b\w+(?:ed|es|s)\b",
    re.IGNORECASE,
)

_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_DATE_MONTH_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"(?<![a-zA-Z])-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?%?")
_MID_SENTENCE_PROPER_NOUN = re.compile(r"(?<!^)\b[A-Z][a-z]{2,}\b")


@dataclass(frozen=True)
class ExtractedClaim:
    """Structured atomic factual claim decomposed from an LLM response."""

    claim_id: str
    question_id: str
    model_id: str
    claim_text: str
    claim_type: str  # "numeric_claim", "temporal_claim", "named_entity_claim", "factual_assertion"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize claim to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ClaimExtractionResult:
    """Complete decomposition output including extracted factual claims and filtered segments."""

    question_id: str
    model_id: str
    raw_response: str
    claims: List[ExtractedClaim] = field(default_factory=list)
    refusal_segments: List[str] = field(default_factory=list)
    non_factual_segments: List[str] = field(default_factory=list)
    status: str = "extracted"  # "extracted", "empty_response", "refusal_only", "no_factual_claims"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize extraction result to a dictionary."""
        return {
            "question_id": self.question_id,
            "model_id": self.model_id,
            "raw_response": self.raw_response,
            "claims": [c.to_dict() for c in self.claims],
            "refusal_segments": list(self.refusal_segments),
            "non_factual_segments": list(self.non_factual_segments),
            "status": self.status,
        }


def split_sentences(text: str) -> List[str]:
    """
    Segment text into sentences while protecting decimal numbers and common abbreviations.
    """
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return []

    protected = _ABBREV_PATTERN.sub(lambda m: m.group(0)[:-1] + _ABBREV_PLACEHOLDER, clean)
    # Protect decimal points between digits (e.g., 7.5, 99.95)
    protected = re.sub(r"(?<=\d)\.(?=\d)", _ABBREV_PLACEHOLDER, protected)

    parts = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [
        p.replace(_ABBREV_PLACEHOLDER, ".").strip()
        for p in parts
        if p and p.replace(_ABBREV_PLACEHOLDER, ".").strip()
    ]
    return sentences


def _looks_like_independent_clause(text: str) -> bool:
    """Return True if a clause has at least two words and contains a verb-like predicate."""
    words = text.strip().split()
    if len(words) < 2:
        return False
    return bool(_CLAUSE_VERB_PATTERN.search(text))


def _split_compound_sentence(sentence: str) -> List[str]:
    """
    Decompose compound sentences on semicolons or clause-level conjunctions
    when both sides form independent factual statements.
    """
    # Strip trailing period/exclamation before splitting, then re-normalize
    semi_parts = [p.strip() for p in re.split(r"\s*;\s*", sentence) if p.strip()]
    clauses: List[str] = []

    conj_pattern = re.compile(
        r"\s*(?:,\s*(?:and|but|while|whereas)\s+|\s+(?:and|while|whereas)\s+)\s*",
        re.IGNORECASE,
    )

    for part in semi_parts:
        # Remove leading discourse connectives such as "However, "
        part = re.sub(r"^(?:however|additionally|furthermore|moreover|meanwhile),?\s+", "", part, flags=re.IGNORECASE)
        sub_parts = conj_pattern.split(part)
        if len(sub_parts) > 1 and all(_looks_like_independent_clause(sp) for sp in sub_parts):
            clauses.extend(sp.strip(" ,.") for sp in sub_parts if sp.strip(" ,."))
        else:
            cleaned = part.strip()
            if cleaned:
                clauses.append(cleaned)

    return clauses


def is_refusal_or_uncertainty_statement(text: str) -> bool:
    """Detect whether a sentence explicitly expresses refusal, insufficiency, or uncertainty."""
    low = (text or "").strip().lower()
    if not low:
        return False
    return any(pattern in low for pattern in _REFUSAL_PATTERNS)


def is_conversational_filler(text: str) -> bool:
    """Detect pure conversational boilerplate without factual assertions."""
    norm = re.sub(r"[^a-z0-9\s]", "", (text or "").strip().lower()).strip()
    if not norm:
        return True
    if norm in _CONVERSATIONAL_FILLER:
        return True
    # Purely subjective opinion markers without verifiable numbers/dates
    if re.match(r"^(?:i think|in my opinion|personally|i believe)\b", norm) and not _NUMBER_PATTERN.search(text):
        return True
    return False


def classify_claim_type(claim_text: str) -> str:
    """Tag an atomic factual claim as numeric, temporal, named_entity, or factual_assertion."""
    has_year_or_date = bool(_YEAR_PATTERN.search(claim_text) or _DATE_MONTH_PATTERN.search(claim_text))
    # Strip 4-digit years before checking non-year numeric measurements/counts
    without_years = _YEAR_PATTERN.sub("", claim_text)
    has_numeric = bool(_NUMBER_PATTERN.search(without_years))

    if has_numeric:
        return "numeric_claim"
    if has_year_or_date:
        return "temporal_claim"
    if _MID_SENTENCE_PROPER_NOUN.search(claim_text):
        return "named_entity_claim"
    return "factual_assertion"


class ClaimExtractor:
    """Decomposes generated answers into typed, atomic factual claims."""

    def extract(
        self,
        question_id: str,
        model_id: str,
        response_text: str,
    ) -> ClaimExtractionResult:
        """Extract atomic factual claims from a model-generated answer."""
        clean_resp = (response_text or "").strip()
        if not clean_resp:
            return ClaimExtractionResult(
                question_id=question_id,
                model_id=model_id,
                raw_response="",
                status="empty_response",
            )

        claims: List[ExtractedClaim] = []
        refusals: List[str] = []
        non_factual: List[str] = []
        claim_counter = 0

        for sentence in split_sentences(clean_resp):
            # Strip non-factual conversational prefixes like "According to the passage,"
            trimmed_sentence = re.sub(
                r"^(?:according to (?:the )?(?:passage|context|text|spec sheet|excerpt|report),?\s*|"
                r"based on (?:the )?(?:passage|context|text),?\s*)",
                "",
                sentence,
                flags=re.IGNORECASE,
            ).strip()

            if not trimmed_sentence:
                continue

            if is_refusal_or_uncertainty_statement(trimmed_sentence):
                refusals.append(sentence)
                continue

            if is_conversational_filler(trimmed_sentence):
                non_factual.append(sentence)
                continue

            for clause in _split_compound_sentence(trimmed_sentence):
                if is_refusal_or_uncertainty_statement(clause):
                    refusals.append(clause)
                    continue
                if is_conversational_filler(clause) or len(clause.strip()) < 2:
                    non_factual.append(clause)
                    continue

                claim_counter += 1
                claim_id = f"{question_id}_C{claim_counter:02d}"
                claims.append(
                    ExtractedClaim(
                        claim_id=claim_id,
                        question_id=question_id,
                        model_id=model_id,
                        claim_text=clause.rstrip("."),
                        claim_type=classify_claim_type(clause),
                    )
                )

        if claims:
            status = "extracted"
        elif refusals:
            status = "refusal_only"
        else:
            status = "no_factual_claims"

        return ClaimExtractionResult(
            question_id=question_id,
            model_id=model_id,
            raw_response=clean_resp,
            claims=claims,
            refusal_segments=refusals,
            non_factual_segments=non_factual,
            status=status,
        )
