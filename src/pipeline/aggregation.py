"""
Model-level and category-level metric aggregation with explicit sample denominators.

Preserves every evaluation dimension separately without collapsing all metrics into
one unexplained score, and documents any sample-population differences across models.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _stat_block(values: Sequence[Optional[float | int]]) -> Dict[str, Any]:
    """Compute mean, std, and exact valid sample count `n`."""
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": round(float(np.mean(clean)), 4),
        "std": round(float(np.std(clean)), 4),
        "n": len(clean),
    }


def _aggregate_subset(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate generation, Phase 5, Phase 6, and Phase 7 metrics for a subset of records."""
    total_q = len(records)
    gen_success = sum(1 for r in records if r.get("generation_status") == "success")
    gen_empty = sum(1 for r in records if r.get("generation_status") == "empty_response")
    gen_error = sum(1 for r in records if r.get("generation_status") == "error")

    # Phase 5: Reference-based metrics
    p5_evaluated = [
        r for r in records
        if r.get("reference_evaluation_status") in {"evaluated", "evaluated_without_semantic"}
    ]
    correctness_labels = Counter(
        str(r.get("correctness_label"))
        for r in p5_evaluated
        if r.get("correctness_label")
    )
    numeric_contradictions = sum(
        1 for r in p5_evaluated if r.get("has_numeric_contradiction") is True
    )

    # Phase 6: Context-grounded faithfulness strictly over context-bearing computed records
    p6_context_computed = [
        r for r in records
        if r.get("has_context") is True and r.get("hallucination_detection_status") == "computed"
    ]
    p6_context_refusals = sum(
        1 for r in records
        if r.get("has_context") is True
        and r.get("hallucination_detection_status") == "appropriate_refusal"
    )
    p6_no_context = sum(1 for r in records if r.get("has_context") is False)
    p6_context_hallucinated = sum(
        1 for r in records
        if r.get("has_context") is True and r.get("hallucination_detected") is True
    )
    p6_no_context_flagged = sum(
        1 for r in records
        if r.get("has_context") is False and r.get("hallucination_detected") is True
    )

    # Phase 7: Judge metrics strictly over valid evaluated judge records
    p7_evaluated = [r for r in records if r.get("judge_evaluation_status") == "evaluated"]
    p7_failed = sum(
        1 for r in records
        if r.get("judge_evaluation_status") not in {None, "disabled", "evaluated"}
    )
    p7_ctx_scores = [
        r.get("judge_contextual_faithfulness_score")
        for r in p7_evaluated
        if isinstance(r.get("judge_contextual_faithfulness_score"), int)
    ]

    return {
        "total_questions": total_q,
        "generation": {
            "successful_count": gen_success,
            "empty_response_count": gen_empty,
            "failed_count": gen_error,
            "response_time_seconds": _stat_block(
                [r.get("generation_time_seconds") for r in records if r.get("generation_status") != "error"]
            ),
        },
        "reference_evaluation": {
            "n_evaluated": len(p5_evaluated),
            "exact_match": _stat_block([r.get("exact_match") for r in p5_evaluated]),
            "normalized_exact_match": _stat_block(
                [r.get("normalized_exact_match") for r in p5_evaluated]
            ),
            "token_precision": _stat_block([r.get("token_precision") for r in p5_evaluated]),
            "token_recall": _stat_block([r.get("token_recall") for r in p5_evaluated]),
            "token_f1": _stat_block([r.get("token_f1") for r in p5_evaluated]),
            "semantic_similarity": _stat_block(
                [r.get("semantic_similarity") for r in p5_evaluated]
            ),
            "correctness_score": _stat_block([r.get("correctness_score") for r in p5_evaluated]),
            "correctness_label_distribution": {
                "counts": dict(correctness_labels),
                "n": len(p5_evaluated),
            },
            "numeric_contradiction_count": numeric_contradictions,
        },
        "faithfulness_and_hallucination": {
            "n_context_computed": len(p6_context_computed),
            "n_context_appropriate_refusal": p6_context_refusals,
            "n_no_context_skipped": p6_no_context,
            "faithfulness_ratio": _stat_block(
                [r.get("faithfulness_ratio") for r in p6_context_computed]
            ),
            "contradiction_rate": _stat_block(
                [r.get("contradiction_rate") for r in p6_context_computed]
            ),
            "unsupported_claim_rate": _stat_block(
                [r.get("unsupported_claim_rate") for r in p6_context_computed]
            ),
            "context_hallucination_rate": _stat_block(
                [r.get("context_hallucination_rate") for r in p6_context_computed]
            ),
            "context_hallucination_detected_count": p6_context_hallucinated,
            "no_context_adversarial_flag_count": p6_no_context_flagged,
        },
        "llm_judge": {
            "n_evaluated": len(p7_evaluated),
            "n_failed_or_skipped": p7_failed,
            "relevance_score": _stat_block([r.get("judge_relevance_score") for r in p7_evaluated]),
            "factual_correctness_score": _stat_block(
                [r.get("judge_factual_correctness_score") for r in p7_evaluated]
            ),
            "completeness_score": _stat_block(
                [r.get("judge_completeness_score") for r in p7_evaluated]
            ),
            "contextual_faithfulness_score": _stat_block(p7_ctx_scores),
            "clarity_score": _stat_block([r.get("judge_clarity_score") for r in p7_evaluated]),
            "overall_score": _stat_block([r.get("judge_overall_score") for r in p7_evaluated]),
            "judge_confidence": _stat_block([r.get("judge_confidence") for r in p7_evaluated]),
            "judge_latency_seconds": _stat_block(
                [r.get("judge_latency_seconds") for r in p7_evaluated]
            ),
        },
    }


def aggregate_pipeline_results(combined_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build model-level and category-level summaries with explicit sample counts (`n`)
    and sample-population parity warnings across compared models.
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for rec in combined_results:
        by_model.setdefault(str(rec["model_id"]), []).append(rec)

    models_summary: Dict[str, Any] = {}
    for model_id, model_recs in by_model.items():
        overall = _aggregate_subset(model_recs)
        by_category: Dict[str, Any] = {}
        cats: Dict[str, List[Dict[str, Any]]] = {}
        for rec in model_recs:
            cats.setdefault(str(rec.get("category") or "Uncategorized"), []).append(rec)
        for cat_name, cat_recs in sorted(cats.items()):
            by_category[cat_name] = _aggregate_subset(cat_recs)

        models_summary[model_id] = {
            **overall,
            "by_category": by_category,
        }

    # Explicit sample-population parity verification across models
    parity_notes: List[str] = []
    if len(models_summary) >= 2:
        model_ids = list(models_summary.keys())
        checks = [
            ("generation.successful_count", lambda m: models_summary[m]["generation"]["successful_count"]),
            ("reference_evaluation.n_evaluated", lambda m: models_summary[m]["reference_evaluation"]["n_evaluated"]),
            ("reference_evaluation.semantic_similarity.n", lambda m: models_summary[m]["reference_evaluation"]["semantic_similarity"]["n"]),
            ("faithfulness_and_hallucination.n_context_computed", lambda m: models_summary[m]["faithfulness_and_hallucination"]["n_context_computed"]),
            ("llm_judge.n_evaluated", lambda m: models_summary[m]["llm_judge"]["n_evaluated"]),
            ("llm_judge.contextual_faithfulness_score.n", lambda m: models_summary[m]["llm_judge"]["contextual_faithfulness_score"]["n"]),
        ]
        for label, getter in checks:
            counts = {m: getter(m) for m in model_ids}
            if len(set(counts.values())) > 1:
                parity_notes.append(
                    f"Sample size mismatch on '{label}' across models: {counts}. "
                    "Do not compare raw averages directly without accounting for different denominators."
                )

    return {
        "models": models_summary,
        "sample_population_parity_notes": parity_notes,
        "metric_notes": {
            "faithfulness": "Calculated strictly over context-bearing questions with extracted factual claims (status='computed').",
            "context_hallucination_rate": "Broad unsupported-by-context metric (contradiction_rate + unsupported_claim_rate); does not prove unsupported claims are factually false.",
            "llm_judge": "Approximate evaluator; judge_confidence is self-reported and not a calibrated probability.",
        },
    }
