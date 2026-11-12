"""
Combined result construction and run-level JSON/CSV persistence.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.evaluation.evaluator import EvaluationRecord
from src.generation.generator import GenerationResult
from src.hallucination.hallucination_detector import HallucinationDetectionRecord
from src.judge.judge_evaluator import JudgeEvaluationRecord
from src.pipeline.aggregation import aggregate_pipeline_results
from src.pipeline.pipeline_config import PipelineConfig


def build_combined_results(
    dataset_df: pd.DataFrame,
    model_ids_in_order: Sequence[str],
    generation_map: Mapping[Tuple[str, str], GenerationResult],
    evaluation_map: Mapping[Tuple[str, str], EvaluationRecord],
    hallucination_map: Mapping[Tuple[str, str], HallucinationDetectionRecord],
    judge_map: Mapping[Tuple[str, str], JudgeEvaluationRecord],
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    """
    Assemble unified question-model records retaining all raw and stage-specific metrics.
    """
    combined: List[Dict[str, Any]] = []

    for _, row in dataset_df.iterrows():
        qid = str(row["question_id"])
        question = str(row.get("question", "") or "")
        context = str(row.get("context", "") or "")
        reference_answer = str(row.get("reference_answer", "") or "")
        category = str(row.get("category", "") or "")
        difficulty = str(row.get("difficulty", "") or "")
        expected_behavior = str(row.get("expected_behavior", "") or "")
        hallucination_test_type = str(row.get("hallucination_test_type", "") or "")

        for model_id in model_ids_in_order:
            key = (model_id, qid)
            gen = generation_map.get(key)
            p5 = evaluation_map.get(key) if config.enable_reference_evaluation else None
            p6 = hallucination_map.get(key) if config.enable_hallucination_detection else None
            p7 = judge_map.get(key) if config.enable_judge else None

            p5_correctness = p5.evaluation_metadata.get("correctness", {}) if p5 else {}

            record: Dict[str, Any] = {
                "question_id": qid,
                "question": question,
                "category": category,
                "difficulty": difficulty,
                "context": context,
                "has_context": len(context.strip()) > 0,
                "reference_answer": reference_answer,
                "expected_behavior": expected_behavior,
                "hallucination_test_type": hallucination_test_type,
                "model_key": gen.model_key if gen else "",
                "model_id": model_id,
                "model_backend": gen.backend if gen else "",
                "generated_answer": gen.generated_answer if gen else "",
                "generation_status": gen.status if gen else "missing",
                "generation_time_seconds": gen.generation_time_seconds if gen else None,
                "generation_error": gen.error_message if gen else "Missing generation record.",
                # Phase 5 Reference-based Evaluation
                "exact_match": p5.exact_match if p5 else None,
                "normalized_exact_match": p5.normalized_exact_match if p5 else None,
                "token_precision": p5.token_precision if p5 else None,
                "token_recall": p5.token_recall if p5 else None,
                "token_f1": p5.token_f1 if p5 else None,
                "semantic_similarity": p5.semantic_similarity if p5 else None,
                "correctness_score": p5_correctness.get("correctness_score") if p5 else None,
                "correctness_label": p5_correctness.get("correctness_label") if p5 else None,
                "key_info_coverage": p5_correctness.get("key_info_coverage") if p5 else None,
                "has_numeric_contradiction": p5_correctness.get("has_numeric_contradiction") if p5 else None,
                "reference_evaluation_status": (
                    p5.evaluation_status if p5 else ("disabled" if not config.enable_reference_evaluation else "missing")
                ),
                # Phase 6 Faithfulness & Hallucination Detection
                "is_counterfactual_context": p6.is_counterfactual_context if p6 else False,
                "supported_claims": p6.faithfulness.supported_count if p6 else None,
                "contradicted_claims": p6.faithfulness.contradicted_count if p6 else None,
                "unsupported_claims": p6.faithfulness.unsupported_count if p6 else None,
                "total_factual_claims": p6.faithfulness.total_factual_claims if p6 else None,
                "faithfulness_ratio": p6.faithfulness.faithfulness_ratio if p6 else None,
                "contradiction_rate": p6.faithfulness.contradiction_rate if p6 else None,
                "unsupported_claim_rate": p6.faithfulness.unsupported_claim_rate if p6 else None,
                "context_hallucination_rate": p6.context_hallucination_rate if p6 else None,
                "hallucination_detected": p6.hallucination_detected if p6 else None,
                "hallucination_types": list(p6.hallucination_types) if p6 else [],
                "hallucination_detection_status": (
                    p6.detection_status if p6 else ("disabled" if not config.enable_hallucination_detection else "missing")
                ),
                "hallucination_explanation": p6.explanation if p6 else None,
                "claims": [c.to_dict() for c in p6.claims] if p6 else [],
                # Phase 7 LLM-as-a-Judge
                "judge_model_id": p7.judge_model_id if p7 else None,
                "judge_prompt_version": p7.prompt_version if p7 else None,
                "judge_relevance_score": p7.relevance_score if p7 else None,
                "judge_factual_correctness_score": p7.factual_correctness_score if p7 else None,
                "judge_completeness_score": p7.completeness_score if p7 else None,
                "judge_contextual_faithfulness_score": p7.contextual_faithfulness_score if p7 else None,
                "judge_clarity_score": p7.clarity_score if p7 else None,
                "judge_overall_score": p7.overall_score if p7 else None,
                "judge_confidence": p7.judge_confidence if p7 else None,
                "judge_evaluation_status": (
                    p7.evaluation_status if p7 else ("disabled" if not config.enable_judge else "missing")
                ),
                "judge_latency_seconds": p7.judge_latency_seconds if p7 else None,
                "judge_error_message": p7.error_message if p7 else None,
                "judge_dimension_explanations": p7.dimension_explanations if p7 else {},
                "raw_judge_response": p7.raw_judge_response if p7 else None,
            }
            combined.append(record)

    return combined


def combined_results_to_dataframe(combined_results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten combined question-model records into a CSV-exportable DataFrame."""
    flat_rows: List[Dict[str, Any]] = []
    for rec in combined_results:
        row = dict(rec)
        row["hallucination_types"] = "; ".join(rec.get("hallucination_types") or [])
        row["claims"] = json.dumps(rec.get("claims") or [], ensure_ascii=False)
        row["judge_dimension_explanations"] = json.dumps(
            rec.get("judge_dimension_explanations") or {}, ensure_ascii=False
        )
        flat_rows.append(row)
    return pd.DataFrame(flat_rows)


def _installed_package_versions() -> Dict[str, Optional[str]]:
    pkgs = (
        "torch",
        "transformers",
        "accelerate",
        "sentence-transformers",
        "huggingface-hub",
        "pandas",
        "numpy",
        "scikit-learn",
        "streamlit",
        "plotly",
        "requests",
        "python-dotenv",
    )
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def save_run_outputs(
    run_id: str,
    run_dir: Path,
    summaries_dir: Path,
    config: PipelineConfig,
    dataset_hash: str,
    model_specs_info: List[Dict[str, Any]],
    combined_results: Sequence[Dict[str, Any]],
    started_at_utc: str,
    resumed_from_checkpoint: bool,
    device_resolved: str,
    generation_skipped_from_checkpoint: int = 0,
) -> Dict[str, Path]:
    """
    Compute final model summaries and write combined_results.json,
    question_level_results.csv, model_summary.json, and run_metadata.json.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    finished_at_utc = datetime.now(timezone.utc).isoformat()

    summary = aggregate_pipeline_results(combined_results)

    disabled_stages: Dict[str, str] = {}
    if not config.enable_reference_evaluation:
        disabled_stages["evaluation_results.json"] = "Disabled via enable_reference_evaluation=False."
    if not config.enable_hallucination_detection:
        disabled_stages["hallucination_results.json"] = "Disabled via enable_hallucination_detection=False."
    if not config.enable_judge:
        disabled_stages["judge_results.json"] = "Disabled via enable_judge=False."

    stage_counts = {
        "total_question_model_pairs": len(combined_results),
        "generation_success": sum(1 for r in combined_results if r["generation_status"] == "success"),
        "generation_empty": sum(1 for r in combined_results if r["generation_status"] == "empty_response"),
        "generation_error": sum(1 for r in combined_results if r["generation_status"] == "error"),
        "generation_skipped_from_checkpoint": generation_skipped_from_checkpoint,
        "reference_evaluated": sum(
            1 for r in combined_results if r["reference_evaluation_status"] in {"evaluated", "evaluated_without_semantic"}
        ),
        "reference_failed": sum(
            1 for r in combined_results if r["reference_evaluation_status"] in {"evaluation_error", "generation_error"}
        ),
        "reference_skipped": sum(
            1 for r in combined_results if r["reference_evaluation_status"] in {"disabled", "empty_generated_answer", "empty_reference_answer"}
        ),
        "hallucination_computed": sum(
            1 for r in combined_results if r["hallucination_detection_status"] == "computed"
        ),
        "hallucination_no_context_evaluated": sum(
            1 for r in combined_results if r["hallucination_detection_status"] == "no_context_evaluated"
        ),
        "hallucination_appropriate_refusal": sum(
            1 for r in combined_results if r["hallucination_detection_status"] == "appropriate_refusal"
        ),
        "hallucination_failed": sum(
            1 for r in combined_results if r["hallucination_detection_status"] in {"detection_error", "generation_error"}
        ),
        "hallucination_skipped": sum(
            1 for r in combined_results if r["hallucination_detection_status"] in {"disabled", "empty_response", "no_factual_claims"}
        ),
        "judge_evaluated": sum(
            1 for r in combined_results if r["judge_evaluation_status"] == "evaluated"
        ),
        "judge_failed": sum(
            1
            for r in combined_results
            if r["judge_evaluation_status"]
            not in {None, "disabled", "evaluated", "empty_generated_answer", "generation_error"}
        ),
        "judge_skipped": sum(
            1
            for r in combined_results
            if r["judge_evaluation_status"] in {"disabled", "empty_generated_answer", "generation_error"}
        ),
    }

    metadata = {
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "dataset_path": str(config.dataset_path),
        "dataset_hash": dataset_hash,
        "models": model_specs_info,
        "prompt_template": {
            "name": config.prompt_template_name,
            "version": config.prompt_template_name,
            "system_prompt": config.system_prompt,
        },
        "generation_config": config.generation_config.to_dict(),
        "evaluation_models": {
            "embedding_model_id": config.embedding_model_id,
            "nli_model_id": config.nli_model_id,
        },
        "evaluation_config": {
            "correctness_config": asdict(config.correctness_config),
            "verification_config": asdict(config.verification_config),
            "retrieval_top_k": config.retrieval_top_k,
            "retrieval_min_similarity": config.retrieval_min_similarity,
        },
        "judge_config": (
            config.judge_config.to_dict()
            if (config.enable_judge and config.judge_config is not None)
            else None
        ),
        "execution_mode": "sequential" if config.sequential_model_execution else "in_memory",
        "resolved_device": device_resolved,
        "enabled_stages": {
            "reference_evaluation": config.enable_reference_evaluation,
            "hallucination_detection": config.enable_hallucination_detection,
            "llm_judge": config.enable_judge,
        },
        "omitted_stage_files": disabled_stages,
        "stage_counts": stage_counts,
        "environment": _installed_package_versions(),
    }

    combined_path = run_dir / "combined_results.json"
    csv_path = run_dir / "question_level_results.csv"
    summary_path = run_dir / "model_summary.json"
    metadata_path = run_dir / "run_metadata.json"
    global_summary_path = summaries_dir / f"{run_id}_summary.json"

    with combined_path.open("w", encoding="utf-8") as fh:
        json.dump(list(combined_results), fh, indent=2, ensure_ascii=False)

    df_flat = combined_results_to_dataframe(combined_results)
    df_flat.to_csv(csv_path, index=False, encoding="utf-8")

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    with global_summary_path.open("w", encoding="utf-8") as fh:
        json.dump({"metadata": metadata, "summary": summary}, fh, indent=2, ensure_ascii=False)

    return {
        "combined_results_json": combined_path,
        "question_level_results_csv": csv_path,
        "model_summary_json": summary_path,
        "run_metadata_json": metadata_path,
        "global_summary_json": global_summary_path,
    }
