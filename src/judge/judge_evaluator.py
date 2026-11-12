"""
LLM-as-a-Judge evaluator orchestrator.

Evaluates single answers, multiple candidate models independently under a
blind, standardized rubric, and integrates cleanly alongside Phases 4, 5, and 6
outputs without mutating or overwriting any prior metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd

from src.evaluation.evaluator import EvaluationRecord
from src.generation.generator import GenerationResult
from src.hallucination.hallucination_detector import HallucinationDetectionRecord
from src.judge.judge_client import (
    BaseJudgeClient,
    JudgeAPIError,
    JudgeClientError,
    JudgeMissingAPIKeyError,
    JudgeRateLimitError,
    JudgeTimeoutError,
    create_judge_client,
)
from src.judge.judge_config import JudgeConfig
from src.judge.judge_metrics import compute_overall_score
from src.judge.judge_parser import parse_and_validate_judge_output
from src.judge.judge_prompt import build_judge_prompt


@dataclass
class JudgeEvaluationRecord:
    """
    Structured LLM-as-a-Judge evaluation record with raw output auditing.

    Note: `judge_confidence` is a self-reported estimate from the judge model
    in [0.0, 1.0]; it is not a calibrated probability or an objective measure of
    answer correctness.
    """

    question_id: str
    evaluated_model_id: str
    judge_model_id: str
    prompt_version: str
    relevance_score: Optional[int]
    factual_correctness_score: Optional[int]
    completeness_score: Optional[int]
    contextual_faithfulness_score: Optional[Union[int, str]]
    clarity_score: Optional[int]
    overall_score: Optional[float]
    dimension_explanations: Dict[str, str]
    judge_confidence: Optional[float]
    evaluation_status: str
    raw_judge_response: str = ""
    judge_latency_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record into a JSON-compatible dictionary."""
        return asdict(self)


class JudgeEvaluator:
    """Independent rubric-based LLM-as-a-Judge orchestrator."""

    def __init__(
        self,
        judge_client: Optional[BaseJudgeClient] = None,
        judge_config: Optional[JudgeConfig] = None,
    ) -> None:
        self.judge_config = judge_config or JudgeConfig()
        self.judge_client: BaseJudgeClient = judge_client or create_judge_client(
            judge_config=self.judge_config
        )

    def evaluate_single(
        self,
        question_id: str,
        evaluated_model_id: str,
        question: str,
        reference_answer: str,
        generated_answer: str,
        context: str = "",
        category: str = "",
        hallucination_test_type: str = "",
        expected_behavior: str = "",
        generation_status: str = "success",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeEvaluationRecord:
        """
        Blindly evaluate a single answer using the standardized rubric.

        Note: `evaluated_model_id` is recorded ONLY on the output record and is
        never sent inside the judge prompt.
        """
        clean_answer = (generated_answer or "").strip()
        clean_context = (context or "").strip()
        has_context = len(clean_context) > 0
        is_counterfactual = (
            hallucination_test_type == "counterfactual_context"
            or expected_behavior == "follow_counterfactual_context"
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        base_meta: Dict[str, Any] = {
            "category": category,
            "hallucination_test_type": hallucination_test_type,
            "expected_behavior": expected_behavior,
            "has_context": has_context,
            "is_counterfactual_context": is_counterfactual,
            "timestamp_utc": timestamp,
        }
        if extra_metadata:
            base_meta.update(extra_metadata)

        if generation_status == "error" or not clean_answer:
            status = "generation_error" if generation_status == "error" else "empty_generated_answer"
            return JudgeEvaluationRecord(
                question_id=question_id,
                evaluated_model_id=evaluated_model_id,
                judge_model_id=self.judge_client.judge_model_id,
                prompt_version=self.judge_config.prompt_version,
                relevance_score=None,
                factual_correctness_score=None,
                completeness_score=None,
                contextual_faithfulness_score=None,
                clarity_score=None,
                overall_score=None,
                dimension_explanations={},
                judge_confidence=None,
                evaluation_status=status,
                raw_judge_response="",
                judge_latency_seconds=0.0,
                error_message="Skipped judging due to missing or failed generator response.",
                metadata=base_meta,
            )

        prompt = build_judge_prompt(
            question_id=question_id,
            question=question,
            reference_answer=reference_answer,
            generated_answer=clean_answer,
            context=clean_context,
            is_counterfactual_context=is_counterfactual,
            prompt_version=self.judge_config.prompt_version,
        )

        try:
            raw_resp = self.judge_client.evaluate_prompt(prompt)
        except JudgeMissingAPIKeyError as exc:
            return self._failed_record(question_id, evaluated_model_id, "missing_api_key", str(exc), "", 0.0, base_meta)
        except JudgeTimeoutError as exc:
            return self._failed_record(question_id, evaluated_model_id, "judge_timeout", str(exc), "", 0.0, base_meta)
        except JudgeRateLimitError as exc:
            return self._failed_record(question_id, evaluated_model_id, "judge_rate_limited", str(exc), "", 0.0, base_meta)
        except (JudgeAPIError, JudgeClientError) as exc:
            return self._failed_record(question_id, evaluated_model_id, "judge_api_error", str(exc), "", 0.0, base_meta)

        parsed = parse_and_validate_judge_output(raw_resp.raw_text, has_context=has_context)
        base_meta["validation_warnings"] = parsed.warnings
        base_meta["overall_summary"] = parsed.overall_summary
        base_meta["judge_backend"] = raw_resp.backend

        if not parsed.is_valid:
            return self._failed_record(
                question_id,
                evaluated_model_id,
                "invalid_judge_output",
                "; ".join(parsed.errors),
                raw_resp.raw_text,
                raw_resp.latency_seconds,
                base_meta,
            )

        overall = compute_overall_score(
            relevance_score=parsed.relevance_score,
            factual_correctness_score=parsed.factual_correctness_score,
            completeness_score=parsed.completeness_score,
            contextual_faithfulness_score=parsed.contextual_faithfulness_score,
            clarity_score=parsed.clarity_score,
            weights=self.judge_config.dimension_weights,
        )

        return JudgeEvaluationRecord(
            question_id=question_id,
            evaluated_model_id=evaluated_model_id,
            judge_model_id=raw_resp.judge_model_id,
            prompt_version=self.judge_config.prompt_version,
            relevance_score=parsed.relevance_score,
            factual_correctness_score=parsed.factual_correctness_score,
            completeness_score=parsed.completeness_score,
            contextual_faithfulness_score=parsed.contextual_faithfulness_score,
            clarity_score=parsed.clarity_score,
            overall_score=overall,
            dimension_explanations=parsed.dimension_explanations,
            judge_confidence=parsed.judge_confidence,
            evaluation_status="evaluated",
            raw_judge_response=raw_resp.raw_text,
            judge_latency_seconds=raw_resp.latency_seconds,
            error_message=None,
            metadata=base_meta,
        )

    def _failed_record(
        self,
        question_id: str,
        evaluated_model_id: str,
        status: str,
        error_msg: str,
        raw_text: str,
        latency: float,
        metadata: Dict[str, Any],
    ) -> JudgeEvaluationRecord:
        return JudgeEvaluationRecord(
            question_id=question_id,
            evaluated_model_id=evaluated_model_id,
            judge_model_id=self.judge_client.judge_model_id,
            prompt_version=self.judge_config.prompt_version,
            relevance_score=None,
            factual_correctness_score=None,
            completeness_score=None,
            contextual_faithfulness_score=None,
            clarity_score=None,
            overall_score=None,
            dimension_explanations={},
            judge_confidence=None,
            evaluation_status=status,
            raw_judge_response=raw_text,
            judge_latency_seconds=latency,
            error_message=error_msg,
            metadata=metadata,
        )

    def evaluate_question_multi_model(
        self,
        question_id: str,
        question: str,
        reference_answer: str,
        model_answers: Mapping[str, str],
        context: str = "",
        category: str = "",
        hallucination_test_type: str = "",
        expected_behavior: str = "",
    ) -> List[JudgeEvaluationRecord]:
        """Evaluate multiple model answers for the same question independently."""
        return [
            self.evaluate_single(
                question_id=question_id,
                evaluated_model_id=model_id,
                question=question,
                reference_answer=reference_answer,
                generated_answer=answer,
                context=context,
                category=category,
                hallucination_test_type=hallucination_test_type,
                expected_behavior=expected_behavior,
            )
            for model_id, answer in model_answers.items()
        ]

    def evaluate_batch(self, items: Sequence[Mapping[str, Any]]) -> List[JudgeEvaluationRecord]:
        """Evaluate a batch of item dictionaries independently."""
        return [
            self.evaluate_single(
                question_id=str(item["question_id"]),
                evaluated_model_id=str(item["evaluated_model_id"]),
                question=str(item.get("question") or ""),
                reference_answer=str(item.get("reference_answer") or ""),
                generated_answer=str(item.get("generated_answer") or ""),
                context=str(item.get("context") or ""),
                category=str(item.get("category") or ""),
                hallucination_test_type=str(item.get("hallucination_test_type") or ""),
                expected_behavior=str(item.get("expected_behavior") or ""),
                generation_status=str(item.get("generation_status") or "success"),
                extra_metadata=item.get("extra_metadata"),
            )
            for item in items
        ]

    def evaluate_from_pipeline_outputs(
        self,
        generation_results: Sequence[Union[GenerationResult, Mapping[str, Any]]],
        dataset_df: pd.DataFrame,
        phase5_records: Optional[Sequence[EvaluationRecord]] = None,
        phase6_records: Optional[Sequence[HallucinationDetectionRecord]] = None,
    ) -> List[JudgeEvaluationRecord]:
        """
        Consume Phase 4 generation results alongside dataset ground truths and optional
        Phase 5 / Phase 6 context metadata without overwriting prior evaluations.
        """
        ds_lookup: Dict[str, Dict[str, str]] = {}
        for _, row in dataset_df.iterrows():
            qid = str(row["question_id"])
            ds_lookup[qid] = {
                "question": str(row.get("question", "") or ""),
                "context": str(row.get("context", "") or ""),
                "reference_answer": str(row.get("reference_answer", "") or ""),
                "category": str(row.get("category", "") or ""),
                "hallucination_test_type": str(row.get("hallucination_test_type", "") or ""),
                "expected_behavior": str(row.get("expected_behavior", "") or ""),
            }

        p5_lookup: Dict[tuple[str, str], EvaluationRecord] = {
            (r.question_id, r.model_id): r for r in (phase5_records or [])
        }
        p6_lookup: Dict[tuple[str, str], HallucinationDetectionRecord] = {
            (r.question_id, r.model_id): r for r in (phase6_records or [])
        }

        items: List[Dict[str, Any]] = []
        for res in generation_results:
            d = res.to_dict() if isinstance(res, GenerationResult) else dict(res)
            qid = str(d["question_id"])
            mid = str(d.get("model_id", ""))
            ds_row = ds_lookup.get(qid, {})

            extra: Dict[str, Any] = {
                "generation_time_seconds": d.get("generation_time_seconds"),
            }
            if (qid, mid) in p5_lookup:
                extra["phase5_semantic_similarity"] = p5_lookup[(qid, mid)].semantic_similarity
                extra["phase5_token_f1"] = p5_lookup[(qid, mid)].token_f1
            if (qid, mid) in p6_lookup:
                extra["phase6_faithfulness_ratio"] = p6_lookup[(qid, mid)].faithfulness.faithfulness_ratio
                extra["phase6_hallucination_detected"] = p6_lookup[(qid, mid)].hallucination_detected

            items.append(
                {
                    "question_id": qid,
                    "evaluated_model_id": mid,
                    "question": ds_row.get("question", str(d.get("question", ""))),
                    "context": ds_row.get("context", str(d.get("context", ""))),
                    "reference_answer": ds_row.get("reference_answer", ""),
                    "generated_answer": str(d.get("generated_answer", "") or ""),
                    "category": ds_row.get("category", ""),
                    "hallucination_test_type": ds_row.get("hallucination_test_type", ""),
                    "expected_behavior": ds_row.get("expected_behavior", ""),
                    "generation_status": str(d.get("status", "success")),
                    "extra_metadata": extra,
                }
            )

        return self.evaluate_batch(items)

    def evaluate_multiple_models(
        self,
        model_results_map: Mapping[str, Sequence[Union[GenerationResult, Mapping[str, Any]]]],
        dataset_df: pd.DataFrame,
        phase5_map: Optional[Mapping[str, Sequence[EvaluationRecord]]] = None,
        phase6_map: Optional[Mapping[str, Sequence[HallucinationDetectionRecord]]] = None,
    ) -> Dict[str, List[JudgeEvaluationRecord]]:
        """Evaluate multiple models independently over identical questions using the same rubric."""
        return {
            model_name: self.evaluate_from_pipeline_outputs(
                generation_results=results,
                dataset_df=dataset_df,
                phase5_records=(phase5_map or {}).get(model_name),
                phase6_records=(phase6_map or {}).get(model_name),
            )
            for model_name, results in model_results_map.items()
        }


def judge_records_to_dataframe(records: Iterable[JudgeEvaluationRecord]) -> pd.DataFrame:
    """Convert JudgeEvaluationRecord items into a flat CSV-friendly DataFrame."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        rows.append(
            {
                "question_id": rec.question_id,
                "evaluated_model_id": rec.evaluated_model_id,
                "judge_model_id": rec.judge_model_id,
                "prompt_version": rec.prompt_version,
                "relevance_score": rec.relevance_score,
                "factual_correctness_score": rec.factual_correctness_score,
                "completeness_score": rec.completeness_score,
                "contextual_faithfulness_score": rec.contextual_faithfulness_score,
                "clarity_score": rec.clarity_score,
                "overall_score": rec.overall_score,
                "judge_confidence": rec.judge_confidence,
                "evaluation_status": rec.evaluation_status,
                "judge_latency_seconds": rec.judge_latency_seconds,
                "error_message": rec.error_message,
                "dimension_explanations": json.dumps(rec.dimension_explanations, ensure_ascii=False),
                "raw_judge_response": rec.raw_judge_response,
                "metadata": json.dumps(rec.metadata, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def save_judge_results(
    records: Sequence[JudgeEvaluationRecord],
    output_dir: Path,
    run_name: str,
) -> Dict[str, Path]:
    """Persist judge evaluation records to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in records], fh, indent=2, ensure_ascii=False)

    df = judge_records_to_dataframe(records)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return {"json": json_path, "csv": csv_path}


def load_judge_results(json_path: Path) -> List[JudgeEvaluationRecord]:
    """Reload JudgeEvaluationRecord objects from a saved JSON file."""
    with json_path.open("r", encoding="utf-8") as fh:
        payload: List[Dict[str, Any]] = json.load(fh)
    return [JudgeEvaluationRecord(**entry) for entry in payload]
