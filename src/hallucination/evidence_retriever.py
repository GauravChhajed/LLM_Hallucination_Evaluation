"""
Context evidence retrieval for claim-level verification.

Segments supplied context passages into sentences and retrieves the most relevant
supporting/candidate context sentences for each extracted factual claim using
semantic similarity and lexical overlap. Never treats low similarity as contradiction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.lexical_metrics import tokenize_text
from src.evaluation.semantic_similarity import (
    BaseEmbeddingProvider,
    SentenceTransformerEmbedder,
    cosine_similarity,
)
from src.hallucination.claim_extractor import split_sentences


@dataclass(frozen=True)
class RetrievedEvidence:
    """A single context sentence matched to a claim with relevance scores."""

    sentence: str
    sentence_index: int
    similarity_score: float
    lexical_overlap: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize retrieved evidence entry."""
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRetrievalResult:
    """Audit-ready retrieval output pairing a claim with retrieved context sentences."""

    claim_id: str
    claim_text: str
    has_context: bool
    context_sentences: List[str] = field(default_factory=list)
    retrieved_evidence: List[RetrievedEvidence] = field(default_factory=list)
    best_evidence: Optional[RetrievedEvidence] = None
    best_similarity: float = 0.0
    retrieval_status: str = "retrieved"  # "retrieved", "no_relevant_evidence", "no_context", "error"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize retrieval result to a dictionary."""
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "has_context": self.has_context,
            "context_sentences": list(self.context_sentences),
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
            "best_evidence": self.best_evidence.to_dict() if self.best_evidence else None,
            "best_similarity": self.best_similarity,
            "retrieval_status": self.retrieval_status,
            "error_message": self.error_message,
        }


def _lexical_jaccard(text_a: str, text_b: str) -> float:
    """Compute token Jaccard overlap between two short texts as a secondary signal."""
    tokens_a = set(tokenize_text(text_a, remove_articles=True))
    tokens_b = set(tokenize_text(text_b, remove_articles=True))
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    return round(len(tokens_a & tokens_b) / len(union), 6)


class EvidenceRetriever:
    """Retrieves top-k candidate evidence sentences from a supplied context."""

    def __init__(
        self,
        embedder: Optional[BaseEmbeddingProvider] = None,
        top_k: int = 2,
        min_similarity_threshold: float = 0.30,
    ) -> None:
        self.embedder: BaseEmbeddingProvider = embedder or SentenceTransformerEmbedder()
        self.top_k = max(1, int(top_k))
        self.min_similarity_threshold = float(min_similarity_threshold)

    def retrieve_for_claims(
        self,
        claims: Sequence[tuple[str, str]],  # (claim_id, claim_text)
        context: str,
    ) -> List[EvidenceRetrievalResult]:
        """
        Retrieve candidate context evidence sentences for a batch of (claim_id, claim_text).
        """
        clean_context = (context or "").strip()
        if not clean_context:
            return [
                EvidenceRetrievalResult(
                    claim_id=cid,
                    claim_text=ctext,
                    has_context=False,
                    retrieval_status="no_context",
                )
                for cid, ctext in claims
            ]

        context_sentences = split_sentences(clean_context)
        if not context_sentences:
            return [
                EvidenceRetrievalResult(
                    claim_id=cid,
                    claim_text=ctext,
                    has_context=False,
                    retrieval_status="no_context",
                )
                for cid, ctext in claims
            ]

        if not claims:
            return []

        claim_texts = [ctext for _, ctext in claims]
        try:
            ctx_embs = self.embedder.encode(context_sentences)
            claim_embs = self.embedder.encode(claim_texts)
        except Exception as exc:
            return [
                EvidenceRetrievalResult(
                    claim_id=cid,
                    claim_text=ctext,
                    has_context=True,
                    context_sentences=context_sentences,
                    retrieval_status="error",
                    error_message=str(exc),
                )
                for cid, ctext in claims
            ]

        results: List[EvidenceRetrievalResult] = []
        for idx, (cid, ctext) in enumerate(claims):
            c_vec = claim_embs[idx]
            scored: List[RetrievedEvidence] = []
            for s_idx, sent in enumerate(context_sentences):
                sim = cosine_similarity(c_vec, ctx_embs[s_idx])
                lex = _lexical_jaccard(ctext, sent)
                scored.append(
                    RetrievedEvidence(
                        sentence=sent,
                        sentence_index=s_idx,
                        similarity_score=sim,
                        lexical_overlap=lex,
                    )
                )

            scored.sort(key=lambda item: (item.similarity_score, item.lexical_overlap), reverse=True)
            best = scored[0] if scored else None
            filtered = [
                item
                for item in scored[: self.top_k]
                if item.similarity_score >= self.min_similarity_threshold
                or item.lexical_overlap >= 0.25
            ]

            status = "retrieved" if filtered else "no_relevant_evidence"
            results.append(
                EvidenceRetrievalResult(
                    claim_id=cid,
                    claim_text=ctext,
                    has_context=True,
                    context_sentences=context_sentences,
                    retrieved_evidence=filtered,
                    best_evidence=best,
                    best_similarity=best.similarity_score if best else 0.0,
                    retrieval_status=status,
                )
            )

        return results

    def retrieve(self, claim_id: str, claim_text: str, context: str) -> EvidenceRetrievalResult:
        """Retrieve evidence sentences for a single claim."""
        return self.retrieve_for_claims([(claim_id, claim_text)], context)[0]
