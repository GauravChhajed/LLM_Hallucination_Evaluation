"""
Judge score aggregation, documented overall weighting, and reliability metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from src.judge.judge_config import DEFAULT_DIMENSION_WEIGHTS
from src.judge.judge_parser import NOT_APPLICABLE_LABEL

if TYPE_CHECKING:
    from src.judge.judge_evaluator import JudgeEvaluationRecord


def compute_overall_score(
    relevance_score: Optional[int],
    factual_correctness_score: Optional[int],
    completeness_score: Optional[int],
    contextual_faithfulness_score: Optional[Union[int, str]],
    clarity_score: Optional[int],
    weights: Optional[Mapping[str, float]] = None,
) -> Optional[float]:
    """
    Compute an explicitly weighted overall score in [1.0, 5.0] across applicable dimensions.

    When `contextual_faithfulness_score == 'NOT_APPLICABLE'`, its weight is excluded and
    the remaining four dimension weights are re-normalized to sum to 1.0.
    """
    active_weights = dict(weights or DEFAULT_DIMENSION_WEIGHTS)
    scores: Dict[str, Optional[int]] = {
        "relevance": relevance_score,
        "factual_correctness": factual_correctness_score,
        "completeness": completeness_score,
        "clarity": clarity_score,
    }
    if isinstance(contextual_faithfulness_score, int):
        scores["contextual_faithfulness"] = contextual_faithfulness_score

    if any(val is None for val in scores.values()):
        return None

    total_weight = sum(active_weights.get(dim, 0.0) for dim in scores)
    if total_weight <= 0.0:
        return None

    weighted_sum = sum(
        float(scores[dim]) * active_weights.get(dim, 0.0) for dim in scores
    )
    return round(weighted_sum / total_weight, 4)


def aggregate_judge_metrics(
    records: Sequence["JudgeEvaluationRecord"],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate judge dimension averages and status counts per evaluated model.

    Failed/invalid judge evaluations are counted explicitly and excluded from means.
    """
    grouped: Dict[str, List["JudgeEvaluationRecord"]] = {}
    for rec in records:
        grouped.setdefault(rec.evaluated_model_id, []).append(rec)

    summary: Dict[str, Dict[str, Any]] = {}
    for model_id, model_recs in grouped.items():
        valid = [r for r in model_recs if r.evaluation_status == "evaluated"]
        failed = [r for r in model_recs if r.evaluation_status != "evaluated"]

        def _mean(values: List[Optional[int | float]]) -> Optional[float]:
            clean = [float(v) for v in values if isinstance(v, (int, float))]
            return round(float(np.mean(clean)), 4) if clean else None

        ctx_scores: List[Optional[int | float]] = [
            r.contextual_faithfulness_score
            for r in valid
            if isinstance(r.contextual_faithfulness_score, int)
        ]

        summary[model_id] = {
            "total_records": len(model_recs),
            "evaluated_count": len(valid),
            "failed_or_skipped_count": len(failed),
            "avg_relevance_score": _mean([r.relevance_score for r in valid]),
            "avg_factual_correctness_score": _mean([r.factual_correctness_score for r in valid]),
            "avg_completeness_score": _mean([r.completeness_score for r in valid]),
            "avg_contextual_faithfulness_score": _mean(ctx_scores),
            "avg_clarity_score": _mean([r.clarity_score for r in valid]),
            "avg_overall_score": _mean([r.overall_score for r in valid]),
            "avg_judge_confidence": _mean([r.judge_confidence for r in valid]),
            "avg_judge_latency_seconds": _mean([r.judge_latency_seconds for r in valid]),
        }

    return summary


def compute_repeated_judging_consistency(
    repeated_records: Sequence["JudgeEvaluationRecord"],
) -> Dict[str, Any]:
    """
    Measure score stability across repeated judging trials of the same answer
    (supports reliability experiments without claiming bias-free judging).
    """
    valid = [r for r in repeated_records if r.evaluation_status == "evaluated"]
    if not valid:
        return {"num_valid_trials": 0, "overall_score_std": None, "dimension_stds": {}}

    dim_fields = {
        "relevance": [r.relevance_score for r in valid],
        "factual_correctness": [r.factual_correctness_score for r in valid],
        "completeness": [r.completeness_score for r in valid],
        "clarity": [r.clarity_score for r in valid],
    }
    ctx_vals = [
        r.contextual_faithfulness_score
        for r in valid
        if isinstance(r.contextual_faithfulness_score, int)
    ]
    if ctx_vals:
        dim_fields["contextual_faithfulness"] = ctx_vals

    dim_stds = {
        dim: round(float(np.std(vals)), 4) for dim, vals in dim_fields.items() if vals
    }
    overall_vals = [r.overall_score for r in valid if r.overall_score is not None]
    return {
        "num_valid_trials": len(valid),
        "overall_score_std": round(float(np.std(overall_vals)), 4) if overall_vals else None,
        "overall_score_range": (
            round(float(max(overall_vals) - min(overall_vals)), 4) if overall_vals else None
        ),
        "dimension_stds": dim_stds,
    }
