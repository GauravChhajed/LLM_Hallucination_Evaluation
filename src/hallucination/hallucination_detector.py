"""
Hallucination detection orchestrator combining claim extraction, evidence retrieval,
claim verification, and faithfulness metrics.

Distinguishes:
- Context-grounded faithfulness & hallucinations (contradicted vs. unsupported claims).
- Counterfactual-context faithfulness (evaluated strictly against supplied fictional context).
- Context-free questions (faithfulness is skipped; adversarial/reference traps are flagged separately
  without treating every unsupported statement as false).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd

from src.evaluation.correctness import (
    detects_refusal_or_correction,
    extract_numeric_tokens,
    is_refusal_or_correction_expected,
)
from src.evaluation.semantic_similarity import BaseEmbeddingProvider, SentenceTransformerEmbedder
from src.generation.generator import GenerationResult
from src.hallucination.claim_extractor import ClaimExtractionResult, ClaimExtractor
from src.hallucination.claim_verifier import (
    LABEL_CONTRADICTED,
    LABEL_NOT_ENOUGH_INFO,
    BaseNLIScorer,
    ClaimVerificationConfig,
    ClaimVerificationResult,
    ClaimVerifier,
)
from src.hallucination.evidence_retriever import EvidenceRetriever
from src.hallucination.faithfulness import FaithfulnessMetrics, compute_faithfulness_metrics


@dataclass
class HallucinationDetectionRecord:
    """
    Complete claim-level faithfulness and hallucination detection output.

    `context_hallucination_rate` combines contradicted and unsupported claims as a
    broad unsupported-by-context measure; it does not imply every unsupported
    claim is factually false.
    """

    question_id: str
    model_id: str
    category: str
    hallucination_test_type: str
    expected_behavior: str
    has_context: bool
    is_counterfactual_context: bool
    generated_answer: str
    context: str
    reference_answer: str
    claims: List[ClaimVerificationResult]
    faithfulness: FaithfulnessMetrics
    hallucination_detected: bool
    hallucination_types: List[str]
    context_hallucination_rate: Optional[float]
    detection_status: str
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record into JSON-friendly nested dictionary."""
        return {
            "question_id": self.question_id,
            "model_id": self.model_id,
            "category": self.category,
            "hallucination_test_type": self.hallucination_test_type,
            "expected_behavior": self.expected_behavior,
            "has_context": self.has_context,
            "is_counterfactual_context": self.is_counterfactual_context,
            "generated_answer": self.generated_answer,
            "context": self.context,
            "reference_answer": self.reference_answer,
            "claims": [c.to_dict() for c in self.claims],
            "faithfulness": self.faithfulness.to_dict(),
            "hallucination_detected": self.hallucination_detected,
            "hallucination_types": list(self.hallucination_types),
            "context_hallucination_rate": self.context_hallucination_rate,
            "detection_status": self.detection_status,
            "explanation": self.explanation,
            "metadata": dict(self.metadata),
        }


class HallucinationDetector:
    """Detects context-grounded hallucinations and adversarial fabrication patterns."""

    def __init__(
        self,
        embedder: Optional[BaseEmbeddingProvider] = None,
        nli_scorer: Optional[BaseNLIScorer] = None,
        claim_extractor: Optional[ClaimExtractor] = None,
        verification_config: Optional[ClaimVerificationConfig] = None,
        retrieval_top_k: int = 2,
        retrieval_min_similarity: float = 0.30,
    ) -> None:
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.evidence_retriever = EvidenceRetriever(
            embedder=self.embedder,
            top_k=retrieval_top_k,
            min_similarity_threshold=retrieval_min_similarity,
        )
        self.claim_verifier = ClaimVerifier(
            nli_scorer=nli_scorer,
            config=verification_config,
        )

    def detect_single(
        self,
        question_id: str,
        model_id: str,
        generated_answer: str,
        context: str = "",
        reference_answer: str = "",
        category: str = "",
        hallucination_test_type: str = "",
        expected_behavior: str = "",
        generation_status: str = "success",
    ) -> HallucinationDetectionRecord:
        """Analyze a single generated answer for faithfulness and hallucination."""
        clean_context = (context or "").strip()
        clean_answer = (generated_answer or "").strip()
        clean_ref = (reference_answer or "").strip()
        has_context = len(clean_context) > 0
        is_counterfactual = (
            hallucination_test_type == "counterfactual_context"
            or expected_behavior == "follow_counterfactual_context"
        )
        timestamp = datetime.now(timezone.utc).isoformat()

        if generation_status == "error":
            faith = compute_faithfulness_metrics(
                question_id, model_id, has_context, [], extraction_status="empty_response"
            )
            return HallucinationDetectionRecord(
                question_id=question_id,
                model_id=model_id,
                category=category,
                hallucination_test_type=hallucination_test_type,
                expected_behavior=expected_behavior,
                has_context=has_context,
                is_counterfactual_context=is_counterfactual,
                generated_answer=clean_answer,
                context=clean_context,
                reference_answer=clean_ref,
                claims=[],
                faithfulness=faith,
                hallucination_detected=False,
                hallucination_types=[],
                context_hallucination_rate=None,
                detection_status="generation_error",
                explanation="Generation failed; hallucination detection skipped.",
                metadata={"timestamp_utc": timestamp},
            )

        extraction: ClaimExtractionResult = self.claim_extractor.extract(
            question_id=question_id,
            model_id=model_id,
            response_text=clean_answer,
        )

        # Case 1: Context-bearing question -> full claim retrieval & verification
        if has_context:
            claim_pairs = [(c.claim_id, c.claim_text) for c in extraction.claims]
            retrievals = self.evidence_retriever.retrieve_for_claims(claim_pairs, clean_context)
            verifications: List[ClaimVerificationResult] = [
                self.claim_verifier.verify(claim, ret)
                for claim, ret in zip(extraction.claims, retrievals)
            ]

            faith = compute_faithfulness_metrics(
                question_id=question_id,
                model_id=model_id,
                has_context=True,
                verification_results=verifications,
                extraction_status=extraction.status,
            )

            h_types: List[str] = []
            notes: List[str] = []

            if faith.contradicted_count > 0:
                h_types.append("context_contradiction")
                if any(c.claim_type in {"numeric_claim", "temporal_claim"} and c.verification_label == LABEL_CONTRADICTED for c in verifications):
                    h_types.append("fabricated_number_or_date")
                notes.append(f"{faith.contradicted_count} claim(s) contradicted supplied context.")

            if faith.unsupported_count > 0:
                h_types.append("unsupported_context_claim")
                notes.append(
                    f"{faith.unsupported_count} claim(s) could not be verified from supplied context."
                )

            # Check unanswerable context traps where the model invented an answer
            if (
                hallucination_test_type == "unanswerable_from_context"
                and extraction.status != "refusal_only"
                and faith.total_factual_claims > 0
                and faith.unsupported_count + faith.contradicted_count > 0
            ):
                if "unanswerable_context_fabrication" not in h_types:
                    h_types.append("unanswerable_context_fabrication")

            if is_counterfactual:
                notes.append(
                    "Evaluated strictly against supplied counterfactual context rather than real-world priors."
                )

            hallucination_detected = len(h_types) > 0
            context_hallucination_rate = faith.context_hallucination_rate

            if not notes:
                if faith.status == "appropriate_refusal":
                    notes.append("Model appropriately acknowledged missing context information.")
                elif faith.status == "computed" and not hallucination_detected:
                    notes.append("All extracted factual claims were supported by supplied context.")
                else:
                    notes.append("No context-grounded hallucinations detected.")

            return HallucinationDetectionRecord(
                question_id=question_id,
                model_id=model_id,
                category=category,
                hallucination_test_type=hallucination_test_type,
                expected_behavior=expected_behavior,
                has_context=True,
                is_counterfactual_context=is_counterfactual,
                generated_answer=clean_answer,
                context=clean_context,
                reference_answer=clean_ref,
                claims=verifications,
                faithfulness=faith,
                hallucination_detected=hallucination_detected,
                hallucination_types=h_types,
                context_hallucination_rate=context_hallucination_rate,
                detection_status=faith.status,
                explanation=" ".join(notes),
                metadata={
                    "extraction_status": extraction.status,
                    "refusal_segments": extraction.refusal_segments,
                    "non_factual_segments": extraction.non_factual_segments,
                    "retrieval_details": [r.to_dict() for r in retrievals],
                    "timestamp_utc": timestamp,
                },
            )

        # Case 2: No supplied context -> skip context faithfulness, check adversarial/reference signals
        verifications_no_ctx: List[ClaimVerificationResult] = [
            ClaimVerificationResult(
                claim_id=c.claim_id,
                question_id=c.question_id,
                model_id=c.model_id,
                claim_text=c.claim_text,
                claim_type=c.claim_type,
                retrieved_evidence=[],
                evidence_scores=[],
                verification_label=LABEL_NOT_ENOUGH_INFO,
                confidence=0.0,
                evidence_strength=0.0,
                explanation="No context supplied; context faithfulness not evaluated.",
                verifier_backend="no_context",
                details={"confidence_interpretation": "not_applicable_no_context"},
            )
            for c in extraction.claims
        ]
        faith_no_ctx = compute_faithfulness_metrics(
            question_id=question_id,
            model_id=model_id,
            has_context=False,
            verification_results=verifications_no_ctx,
            extraction_status=extraction.status,
        )

        no_ctx_types: List[str] = []
        no_ctx_notes: List[str] = [
            "No context supplied; context-faithfulness was skipped."
        ]

        refusal_expected = is_refusal_or_correction_expected(
            clean_ref,
            expected_behavior=expected_behavior,
            hallucination_test_type=hallucination_test_type,
        )
        refusal_seen = detects_refusal_or_correction(clean_answer) or extraction.status == "refusal_only"

        if clean_answer and refusal_expected and not refusal_seen:
            if hallucination_test_type == "fabricated_citation_or_number":
                no_ctx_types.append("possible_fabricated_citation_or_fact")
            elif hallucination_test_type == "false_premise":
                no_ctx_types.append("false_premise_acceptance")
            else:
                no_ctx_types.append("unwarranted_certainty_without_refusal")
            no_ctx_notes.append(
                "Prompt expected refusal, uncertainty, or false-premise rejection, "
                "which was not detected in the response."
            )
        elif clean_answer and clean_ref:
            ref_nums = extract_numeric_tokens(clean_ref)
            gen_nums = extract_numeric_tokens(clean_answer)
            if ref_nums and gen_nums and len(ref_nums & gen_nums) < len(ref_nums) and len(gen_nums - ref_nums) > 0:
                no_ctx_types.append("reference_numeric_error")
                no_ctx_notes.append(
                    f"Reference-based numeric error: expected {sorted(ref_nums)}, found {sorted(gen_nums)}."
                )

        return HallucinationDetectionRecord(
            question_id=question_id,
            model_id=model_id,
            category=category,
            hallucination_test_type=hallucination_test_type,
            expected_behavior=expected_behavior,
            has_context=False,
            is_counterfactual_context=False,
            generated_answer=clean_answer,
            context="",
            reference_answer=clean_ref,
            claims=verifications_no_ctx,
            faithfulness=faith_no_ctx,
            hallucination_detected=len(no_ctx_types) > 0,
            hallucination_types=no_ctx_types,
            context_hallucination_rate=None,
            detection_status="no_context_evaluated" if clean_answer else "empty_response",
            explanation=" ".join(no_ctx_notes),
            metadata={
                "extraction_status": extraction.status,
                "refusal_segments": extraction.refusal_segments,
                "non_factual_segments": extraction.non_factual_segments,
                "timestamp_utc": timestamp,
            },
        )

    def detect_batch(self, items: Sequence[Mapping[str, Any]]) -> List[HallucinationDetectionRecord]:
        """Run hallucination detection across a sequence of item dictionaries."""
        return [
            self.detect_single(
                question_id=str(item["question_id"]),
                model_id=str(item["model_id"]),
                generated_answer=str(item.get("generated_answer") or ""),
                context=str(item.get("context") or ""),
                reference_answer=str(item.get("reference_answer") or ""),
                category=str(item.get("category") or ""),
                hallucination_test_type=str(item.get("hallucination_test_type") or ""),
                expected_behavior=str(item.get("expected_behavior") or ""),
                generation_status=str(item.get("generation_status") or "success"),
            )
            for item in items
        ]

    def detect_from_generation_results(
        self,
        generation_results: Sequence[Union[GenerationResult, Mapping[str, Any]]],
        dataset_df: pd.DataFrame,
    ) -> List[HallucinationDetectionRecord]:
        """Run detection on Phase 4 GenerationResult outputs joined with dataset metadata."""
        lookup: Dict[str, Dict[str, str]] = {}
        for _, row in dataset_df.iterrows():
            qid = str(row["question_id"])
            lookup[qid] = {
                "context": str(row.get("context", "") or ""),
                "reference_answer": str(row.get("reference_answer", "") or ""),
                "category": str(row.get("category", "") or ""),
                "hallucination_test_type": str(row.get("hallucination_test_type", "") or ""),
                "expected_behavior": str(row.get("expected_behavior", "") or ""),
            }

        items: List[Dict[str, Any]] = []
        for res in generation_results:
            res_dict = res.to_dict() if isinstance(res, GenerationResult) else dict(res)
            qid = str(res_dict["question_id"])
            row_info = lookup.get(qid, {})
            items.append(
                {
                    "question_id": qid,
                    "model_id": str(res_dict.get("model_id", "")),
                    "generated_answer": str(res_dict.get("generated_answer", "") or ""),
                    "context": row_info.get("context", str(res_dict.get("context", "") or "")),
                    "reference_answer": row_info.get("reference_answer", ""),
                    "category": row_info.get("category", ""),
                    "hallucination_test_type": row_info.get("hallucination_test_type", ""),
                    "expected_behavior": row_info.get("expected_behavior", ""),
                    "generation_status": str(res_dict.get("status", "success")),
                }
            )

        return self.detect_batch(items)


def hallucination_records_to_dataframe(
    records: Iterable[HallucinationDetectionRecord],
) -> pd.DataFrame:
    """Convert HallucinationDetectionRecord objects into a flat CSV-friendly DataFrame."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        rows.append(
            {
                "question_id": rec.question_id,
                "model_id": rec.model_id,
                "category": rec.category,
                "hallucination_test_type": rec.hallucination_test_type,
                "has_context": rec.has_context,
                "is_counterfactual_context": rec.is_counterfactual_context,
                "supported_claims": rec.faithfulness.supported_count,
                "contradicted_claims": rec.faithfulness.contradicted_count,
                "unsupported_claims": rec.faithfulness.unsupported_count,
                "total_factual_claims": rec.faithfulness.total_factual_claims,
                "faithfulness_ratio": rec.faithfulness.faithfulness_ratio,
                "contradiction_rate": rec.faithfulness.contradiction_rate,
                "unsupported_claim_rate": rec.faithfulness.unsupported_claim_rate,
                "context_hallucination_rate": rec.context_hallucination_rate,
                "hallucination_detected": rec.hallucination_detected,
                "hallucination_types": "; ".join(rec.hallucination_types),
                "detection_status": rec.detection_status,
                "explanation": rec.explanation,
                "claims_json": json.dumps([c.to_dict() for c in rec.claims], ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def save_hallucination_results(
    records: Sequence[HallucinationDetectionRecord],
    output_dir: Path,
    run_name: str,
) -> Dict[str, Path]:
    """Persist claim-level hallucination detection records to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in records], fh, indent=2, ensure_ascii=False)

    df = hallucination_records_to_dataframe(records)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return {"json": json_path, "csv": csv_path}


def load_hallucination_results(json_path: Path) -> List[HallucinationDetectionRecord]:
    """Reload HallucinationDetectionRecord objects from a saved JSON report."""
    with json_path.open("r", encoding="utf-8") as fh:
        payload: List[Dict[str, Any]] = json.load(fh)

    records: List[HallucinationDetectionRecord] = []
    for entry in payload:
        claims = [ClaimVerificationResult(**c) for c in entry.get("claims", [])]
        faith = FaithfulnessMetrics(**entry["faithfulness"])
        records.append(
            HallucinationDetectionRecord(
                question_id=entry["question_id"],
                model_id=entry["model_id"],
                category=entry.get("category", ""),
                hallucination_test_type=entry.get("hallucination_test_type", ""),
                expected_behavior=entry.get("expected_behavior", ""),
                has_context=entry["has_context"],
                is_counterfactual_context=entry.get("is_counterfactual_context", False),
                generated_answer=entry.get("generated_answer", ""),
                context=entry.get("context", ""),
                reference_answer=entry.get("reference_answer", ""),
                claims=claims,
                faithfulness=faith,
                hallucination_detected=entry["hallucination_detected"],
                hallucination_types=entry.get("hallucination_types", []),
                context_hallucination_rate=entry.get(
                    "context_hallucination_rate", entry.get("hallucination_rate")
                ),
                detection_status=entry["detection_status"],
                explanation=entry.get("explanation", ""),
                metadata=entry.get("metadata", {}),
            )
        )
    return records
