"""
Lexical evaluation metrics: Exact Match, Normalized Exact Match, and Token Precision/Recall/F1.

Normalization preserves meaningful numeric values (decimals, negative signs, fractions),
dates, names, and units while normalizing case, whitespace, external punctuation, and
optional English articles.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence

ENGLISH_ARTICLES = frozenset({"a", "an", "the"})

# Matches alphanumeric tokens, keeping internal decimals (7.5), fractions (1/6),
# negative numbers (-6), and hyphenated identifiers/dates (2024-03, syn-ack).
_TOKEN_PATTERN = re.compile(r"-?\d+(?:[./]\d+)*|[a-z0-9]+(?:[-'/][a-z0-9]+)*", re.IGNORECASE)


@dataclass(frozen=True)
class LexicalMetricsResult:
    """Structured lexical evaluation metrics for a single answer-reference pair."""

    exact_match: float
    normalized_exact_match: float
    token_precision: float
    token_recall: float
    token_f1: float
    generated_token_count: int
    reference_token_count: int
    common_token_count: int


def normalize_text(
    text: str,
    remove_articles: bool = True,
    strip_thousands_commas: bool = True,
) -> str:
    """
    Normalize text for fair lexical comparison without erasing numbers or units.

    Steps:
    1. Strip outer whitespace and lowercase.
    2. Optionally remove thousands-separator commas inside numbers (e.g., 1,000 -> 1000).
    3. Extract normalized tokens preserving decimals (7.5), fractions (1/6), signs (-6),
       and hyphenated words/dates.
    4. Optionally drop standalone English articles ('a', 'an', 'the').
    5. Rejoin tokens with single spaces.
    """
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return ""

    if strip_thousands_commas:
        cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", cleaned)

    tokens = _TOKEN_PATTERN.findall(cleaned)
    if remove_articles:
        tokens = [tok for tok in tokens if tok not in ENGLISH_ARTICLES]

    return " ".join(tokens)


def tokenize_text(text: str, remove_articles: bool = False) -> List[str]:
    """
    Tokenize text into normalized lowercase lexical units.

    Retains numbers (including decimals, signs, and fractions), dates, names,
    and units while removing surrounding punctuation.
    """
    norm = normalize_text(text, remove_articles=remove_articles)
    return norm.split() if norm else []


def compute_exact_match(generated: str, reference: str) -> float:
    """
    Compute strict exact string match after stripping outer whitespace.

    Returns 1.0 if both strings are non-empty and identical, else 0.0.
    """
    gen = (generated or "").strip()
    ref = (reference or "").strip()
    if not gen or not ref:
        return 0.0
    return 1.0 if gen == ref else 0.0


def compute_normalized_exact_match(
    generated: str,
    reference: str,
    remove_articles: bool = True,
) -> float:
    """
    Compute exact match after normalization (case, whitespace, punctuation, articles).

    Returns 1.0 if normalized non-empty strings match, else 0.0.
    """
    norm_gen = normalize_text(generated, remove_articles=remove_articles)
    norm_ref = normalize_text(reference, remove_articles=remove_articles)
    if not norm_gen or not norm_ref:
        return 0.0
    return 1.0 if norm_gen == norm_ref else 0.0


def compute_token_f1_from_tokens(
    generated_tokens: Sequence[str],
    reference_tokens: Sequence[str],
) -> tuple[float, float, float, int]:
    """
    Compute multiset (bag-of-tokens) Precision, Recall, and F1.

    Using Counter intersection properly penalizes repeated tokens in generated answers.
    Returns (precision, recall, f1, common_count).
    """
    if not generated_tokens or not reference_tokens:
        return 0.0, 0.0, 0.0, 0

    gen_counts = Counter(generated_tokens)
    ref_counts = Counter(reference_tokens)
    common = sum((gen_counts & ref_counts).values())

    if common == 0:
        return 0.0, 0.0, 0.0, 0

    precision = common / len(generated_tokens)
    recall = common / len(reference_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return (
        round(float(precision), 6),
        round(float(recall), 6),
        round(float(f1), 6),
        int(common),
    )


def compute_lexical_metrics(
    generated: str,
    reference: str,
    remove_articles: bool = True,
) -> LexicalMetricsResult:
    """Compute Exact Match, Normalized Exact Match, and Token Precision/Recall/F1."""
    em = compute_exact_match(generated, reference)
    nem = compute_normalized_exact_match(generated, reference, remove_articles=remove_articles)

    gen_tokens = tokenize_text(generated, remove_articles=remove_articles)
    ref_tokens = tokenize_text(reference, remove_articles=remove_articles)
    precision, recall, f1, common = compute_token_f1_from_tokens(gen_tokens, ref_tokens)

    return LexicalMetricsResult(
        exact_match=em,
        normalized_exact_match=nem,
        token_precision=precision,
        token_recall=recall,
        token_f1=f1,
        generated_token_count=len(gen_tokens),
        reference_token_count=len(ref_tokens),
        common_token_count=common,
    )
