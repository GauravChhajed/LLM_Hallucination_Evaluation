"""
Unified reference-based evaluation orchestrator.

Consumes Phase 4 GenerationResult outputs alongside dataset ground truths,
computes lexical, semantic similarity, and reference correctness signals,
and persists structured evaluation reports in CSV and JSON formats.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd

from src.evaluation.correctness import CorrectnessAssessment, ReferenceCorrectnessEvaluator
from src.evaluation.lexical_metrics import LexicalMetricsResult, compute_lexical_metrics
from src.evaluation.semantic_similarity import (
    BaseEmbeddingProvider,
    SemanticSimilarityEvaluator,
    SemanticSimilarityResult,
)
from src.generation.generator import GenerationResult


@dataclass
class EvaluationRecord:
    """Standardized Phase 5 evaluation record for a single question-model output."""

    question_id: str
    model_id: str
    generated_answer: str
    reference_answer: str
    exact_match: float
    normalized_exact_match: float
    token_precision: float
    token_recall: float
    token_f1: float
    semantic_similarity: Optional[float]
    evaluation_status: str
    evaluation_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record into a dictionary."""
        return asdict(self)


class ResponseEvaluator:
    """
    Evaluates generated LLM responses against reference answers across multiple models.

    Preserves all individual metric scores alongside transparent correctness diagnostics.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbeddingProvider] = None,
        correctness_evaluator: Optional[ReferenceCorrectnessEvaluator] = None,
        remove_articles: bool = True,
    ) -> None:
        self.semantic_evaluator = SemanticSimilarityEvaluator(embedder=embedder)
        self.correctness_evaluator = correctness_evaluator or ReferenceCorrectnessEvaluator()
        self.remove_articles = remove_articles

    def evaluate_single(
        self,
        question_id: str,
        model_id: str,
        generated_answer: str,
        reference_answer: str,
        generation_status: str = "success",
        question_type: Optional[str] = None,
        expected_behavior: Optional[str] = None,
        hallucination_test_type: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationRecord:
        """Evaluate a single generated answer against its reference answer."""
        item = {
            "question_id": question_id,
            "model_id": model_id,
            "generated_answer": generated_answer,
            "reference_answer": reference_answer,
            "generation_status": generation_status,
            "question_type": question_type,
            "expected_behavior": expected_behavior,
            "hallucination_test_type": hallucination_test_type,
            "extra_metadata": extra_metadata or {},
        }
        return self.evaluate_batch([item])[0]

    def evaluate_batch(
        self,
        items: Sequence[Mapping[str, Any]],
        batch_size: int = 32,
    ) -> List[EvaluationRecord]:
        """
        Evaluate a batch of response-reference dictionaries efficiently.

        Batches semantic embedding generation across valid non-empty pairs.
        """
        if not items:
            return []

        generated_texts: List[str] = []
        reference_texts: List[str] = []
        lexical_results: List[LexicalMetricsResult] = []
        base_statuses: List[str] = []

        for item in items:
            gen_raw = str(item.get("generated_answer") or "")
            ref_raw = str(item.get("reference_answer") or "")
            gen_status = str(item.get("generation_status") or "success")

            clean_gen = gen_raw.strip()
            clean_ref = ref_raw.strip()

            if gen_status == "error":
                base_status = "generation_error"
            elif not clean_ref:
                base_status = "empty_reference_answer"
            elif not clean_gen:
                base_status = "empty_generated_answer"
            else:
                base_status = "evaluated"

            base_statuses.append(base_status)

            if base_status == "evaluated":
                lex = compute_lexical_metrics(
                    clean_gen, clean_ref, remove_articles=self.remove_articles
                )
                generated_texts.append(clean_gen)
                reference_texts.append(clean_ref)
            else:
                lex = LexicalMetricsResult(
                    exact_match=0.0,
                    normalized_exact_match=0.0,
                    token_precision=0.0,
                    token_recall=0.0,
                    token_f1=0.0,
                    generated_token_count=0,
                    reference_token_count=len(clean_ref.split()) if clean_ref else 0,
                    common_token_count=0,
                )
                generated_texts.append("")
                reference_texts.append("")

            lexical_results.append(lex)

        semantic_results: List[SemanticSimilarityResult] = (
            self.semantic_evaluator.compute_batch_similarity(
                generated_texts,
                reference_texts,
                batch_size=batch_size,
            )
        )

        timestamp = datetime.now(timezone.utc).isoformat()
        records: List[EvaluationRecord] = []

        for item, lex, sem, base_status in zip(
            items, lexical_results, semantic_results, base_statuses
        ):
            clean_gen = str(item.get("generated_answer") or "").strip()
            clean_ref = str(item.get("reference_answer") or "").strip()
            q_type = item.get("question_type")
            exp_behavior = item.get("expected_behavior")
            test_type = item.get("hallucination_test_type")

            final_status = base_status
            if base_status == "evaluated" and sem.status == "error":
                final_status = "evaluated_without_semantic"

            correctness: CorrectnessAssessment = self.correctness_evaluator.assess(
                generated_answer=clean_gen if base_status == "evaluated" else "",
                reference_answer=clean_ref,
                lexical=lex,
                semantic_similarity=sem.score,
                question_type=q_type,
                expected_behavior=exp_behavior,
                hallucination_test_type=test_type,
            )

            metadata: Dict[str, Any] = {
                "question_type": q_type,
                "expected_behavior": exp_behavior,
                "hallucination_test_type": test_type,
                "embedding_model": sem.embedding_model,
                "semantic_status": sem.status,
                "semantic_error": sem.error_message,
                "lexical_counts": {
                    "generated_tokens": lex.generated_token_count,
                    "reference_tokens": lex.reference_token_count,
                    "common_tokens": lex.common_token_count,
                },
                "correctness": correctness.to_dict(),
                "timestamp_utc": timestamp,
            }
            if item.get("extra_metadata"):
                metadata.update(dict(item["extra_metadata"]))

            records.append(
                EvaluationRecord(
                    question_id=str(item["question_id"]),
                    model_id=str(item["model_id"]),
                    generated_answer=clean_gen,
                    reference_answer=clean_ref,
                    exact_match=lex.exact_match,
                    normalized_exact_match=lex.normalized_exact_match,
                    token_precision=lex.token_precision,
                    token_recall=lex.token_recall,
                    token_f1=lex.token_f1,
                    semantic_similarity=sem.score,
                    evaluation_status=final_status,
                    evaluation_metadata=metadata,
                )
            )

        return records

    def evaluate_generation_results(
        self,
        generation_results: Sequence[Union[GenerationResult, Mapping[str, Any]]],
        dataset_df: pd.DataFrame,
        batch_size: int = 32,
    ) -> List[EvaluationRecord]:
        """
        Evaluate Phase 4 GenerationResult outputs joined against a reference dataset DataFrame.
        """
        ref_lookup: Dict[str, Dict[str, Any]] = {}
        for _, row in dataset_df.iterrows():
            qid = str(row["question_id"])
            ref_lookup[qid] = {
                "reference_answer": str(row.get("reference_answer", "") or ""),
                "category": str(row.get("category", "") or ""),
                "difficulty": str(row.get("difficulty", "") or ""),
                "expected_behavior": str(row.get("expected_behavior", "") or ""),
                "hallucination_test_type": str(row.get("hallucination_test_type", "") or ""),
            }

        items: List[Dict[str, Any]] = []
        for res in generation_results:
            res_dict = res.to_dict() if isinstance(res, GenerationResult) else dict(res)
            qid = str(res_dict["question_id"])
            ref_info = ref_lookup.get(qid, {})

            items.append(
                {
                    "question_id": qid,
                    "model_id": str(res_dict.get("model_id", "")),
                    "generated_answer": str(res_dict.get("generated_answer", "") or ""),
                    "reference_answer": ref_info.get("reference_answer", ""),
                    "generation_status": str(res_dict.get("status", "success")),
                    "question_type": ref_info.get("category"),
                    "expected_behavior": ref_info.get("expected_behavior"),
                    "hallucination_test_type": ref_info.get("hallucination_test_type"),
                    "extra_metadata": {
                        "model_key": res_dict.get("model_key"),
                        "difficulty": ref_info.get("difficulty"),
                        "generation_time_seconds": res_dict.get("generation_time_seconds"),
                        "generation_error": res_dict.get("error_message"),
                        "missing_reference_in_dataset": qid not in ref_lookup,
                    },
                }
            )

        return self.evaluate_batch(items, batch_size=batch_size)

    def evaluate_multiple_models(
        self,
        model_results_map: Mapping[str, Sequence[Union[GenerationResult, Mapping[str, Any]]]],
        dataset_df: pd.DataFrame,
        batch_size: int = 32,
    ) -> Dict[str, List[EvaluationRecord]]:
        """Evaluate outputs for multiple models against the same reference dataset."""
        return {
            model_name: self.evaluate_generation_results(
                generation_results=results,
                dataset_df=dataset_df,
                batch_size=batch_size,
            )
            for model_name, results in model_results_map.items()
        }


def evaluation_records_to_dataframe(records: Iterable[EvaluationRecord]) -> pd.DataFrame:
    """Convert EvaluationRecord instances into a flat, export-ready DataFrame."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        correctness = rec.evaluation_metadata.get("correctness", {})
        rows.append(
            {
                "question_id": rec.question_id,
                "model_id": rec.model_id,
                "generated_answer": rec.generated_answer,
                "reference_answer": rec.reference_answer,
                "exact_match": rec.exact_match,
                "normalized_exact_match": rec.normalized_exact_match,
                "token_precision": rec.token_precision,
                "token_recall": rec.token_recall,
                "token_f1": rec.token_f1,
                "semantic_similarity": rec.semantic_similarity,
                "correctness_score": correctness.get("correctness_score"),
                "correctness_label": correctness.get("correctness_label"),
                "key_info_coverage": correctness.get("key_info_coverage"),
                "has_numeric_contradiction": correctness.get("has_numeric_contradiction"),
                "evaluation_status": rec.evaluation_status,
                "evaluation_metadata": json.dumps(rec.evaluation_metadata, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def save_evaluation_results(
    records: Sequence[EvaluationRecord],
    output_dir: Path,
    run_name: str,
) -> Dict[str, Path]:
    """Persist evaluation records to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in records], fh, indent=2, ensure_ascii=False)

    df = evaluation_records_to_dataframe(records)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    return {"json": json_path, "csv": csv_path}


def load_evaluation_results(json_path: Path) -> List[EvaluationRecord]:
    """Reload EvaluationRecord objects from a saved JSON file."""
    with json_path.open("r", encoding="utf-8") as fh:
        payload: List[Dict[str, Any]] = json.load(fh)
    return [EvaluationRecord(**entry) for entry in payload]
