"""
Dashboard data-loading, discovery, filtering, and export preparation utilities.

Reads Phase 8 run directories dynamically without mutating pipeline outputs,
handles incomplete or corrupted runs gracefully, and reuses Phase 8 aggregation
logic so metric definitions and denominators stay identical across CLI and UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.pipeline.aggregation import aggregate_pipeline_results
from src.pipeline.result_manager import combined_results_to_dataframe
from src.utils.config import get_config


@dataclass
class RunDescriptor:
    """Lightweight summary descriptor for a discovered run directory."""

    run_id: str
    run_dir: Path
    started_at_utc: str
    finished_at_utc: str
    status: str  # "complete", "incomplete", "corrupted"
    models: List[str] = field(default_factory=list)
    dataset_hash: str = ""
    total_pairs: int = 0
    judge_enabled: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class LoadedRunData:
    """Loaded artifacts and normalized records for a single evaluation run."""

    descriptor: RunDescriptor
    combined_results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    metadata: Dict[str, Any]
    run_config: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def _safe_read_json(path: Path, warnings: List[str], label: str) -> Optional[Any]:
    """Read a JSON file safely, appending human-readable diagnostics on failure."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        warnings.append(f"Could not parse {label} ({path.name}): {exc}")
        return None


def _extract_model_ids_from_config(run_cfg: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
    if isinstance(metadata.get("models"), list):
        ids = [str(m.get("model_id") or m.get("key") or "") for m in metadata["models"] if isinstance(m, dict)]
        ids = [m for m in ids if m]
        if ids:
            return ids
    fp = run_cfg.get("fingerprint") or {}
    if isinstance(fp.get("models"), list):
        ids = [str(m.get("model_id") or m.get("key") or "") for m in fp["models"] if isinstance(m, dict)]
        ids = [m for m in ids if m]
        if ids:
            return ids
    cfg = run_cfg.get("config") or {}
    if isinstance(cfg.get("models"), list):
        return [str(m) for m in cfg["models"]]
    return []


def discover_runs(runs_dir: Optional[Path] = None) -> List[RunDescriptor]:
    """
    Scan `results/runs/` dynamically and return descriptors sorted newest first.
    """
    target_dir = Path(runs_dir) if runs_dir is not None else get_config().paths.results_dir / "runs"
    if not target_dir.exists():
        return []

    descriptors: List[RunDescriptor] = []
    for entry in target_dir.iterdir():
        if not entry.is_dir():
            continue

        warnings: List[str] = []
        metadata_raw = _safe_read_json(entry / "run_metadata.json", warnings, "run_metadata.json")
        run_cfg_raw = _safe_read_json(entry / "run_config.json", warnings, "run_config.json")
        combined_exists = (entry / "combined_results.json").exists() or (
            entry / "question_level_results.csv"
        ).exists()

        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
        run_cfg = run_cfg_raw if isinstance(run_cfg_raw, dict) else {}

        if warnings:
            status = "corrupted"
        elif metadata and combined_exists:
            status = "complete"
        elif run_cfg or (entry / "generation_results.json").exists():
            status = "incomplete"
            warnings.append("Run has partial checkpoints but no finalized run_metadata.json.")
        else:
            # Skip unrelated empty directories with no run artifacts
            continue

        started_at = str(metadata.get("started_at_utc") or "")
        finished_at = str(metadata.get("finished_at_utc") or "")
        if not started_at:
            # Fallback to directory mtime ISO string for incomplete runs
            mtime = pd.Timestamp(entry.stat().st_mtime, unit="s", tz="UTC")
            started_at = mtime.isoformat()

        models = _extract_model_ids_from_config(run_cfg, metadata)
        dataset_hash = str(
            metadata.get("dataset_hash")
            or (run_cfg.get("fingerprint") or {}).get("dataset_hash")
            or ""
        )
        total_pairs = int((metadata.get("stage_counts") or {}).get("total_question_model_pairs") or 0)
        judge_enabled = bool(
            (metadata.get("enabled_stages") or {}).get("llm_judge")
            or (run_cfg.get("config") or {}).get("enable_judge")
            or (entry / "judge_results.json").exists()
        )

        descriptors.append(
            RunDescriptor(
                run_id=entry.name,
                run_dir=entry,
                started_at_utc=started_at,
                finished_at_utc=finished_at,
                status=status,
                models=models,
                dataset_hash=dataset_hash,
                total_pairs=total_pairs,
                judge_enabled=judge_enabled,
                warnings=warnings,
            )
        )

    descriptors.sort(key=lambda d: (d.started_at_utc, d.run_id), reverse=True)
    return descriptors


def _reconstruct_partial_records(run_dir: Path, warnings: List[str]) -> List[Dict[str, Any]]:
    """
    Reconstruct dashboard-compatible records from incremental checkpoint files
    when an interrupted run lacks `combined_results.json`.
    """
    gen_list = _safe_read_json(run_dir / "generation_results.json", warnings, "generation_results.json")
    if not isinstance(gen_list, list):
        return []

    eval_list = _safe_read_json(run_dir / "evaluation_results.json", warnings, "evaluation_results.json")
    hall_list = _safe_read_json(run_dir / "hallucination_results.json", warnings, "hallucination_results.json")
    judge_list = _safe_read_json(run_dir / "judge_results.json", warnings, "judge_results.json")

    eval_map = {
        (str(item.get("model_id", "")), str(item.get("question_id", ""))): item
        for item in (eval_list if isinstance(eval_list, list) else [])
        if isinstance(item, dict)
    }
    hall_map = {
        (str(item.get("model_id", "")), str(item.get("question_id", ""))): item
        for item in (hall_list if isinstance(hall_list, list) else [])
        if isinstance(item, dict)
    }
    judge_map = {
        (str(item.get("evaluated_model_id", "")), str(item.get("question_id", ""))): item
        for item in (judge_list if isinstance(judge_list, list) else [])
        if isinstance(item, dict)
    }

    records: List[Dict[str, Any]] = []
    for gen in gen_list:
        if not isinstance(gen, dict):
            continue
        qid = str(gen.get("question_id", ""))
        mid = str(gen.get("model_id", ""))
        key = (mid, qid)
        p5 = eval_map.get(key)
        p6 = hall_map.get(key)
        p7 = judge_map.get(key)
        p5_meta = (p5.get("evaluation_metadata") or {}) if isinstance(p5, dict) else {}
        p5_corr = p5_meta.get("correctness") or {}
        p6_faith = (p6.get("faithfulness") or {}) if isinstance(p6, dict) else {}
        ctx = str(gen.get("context") or (p6.get("context") if isinstance(p6, dict) else "") or "")

        records.append(
            {
                "question_id": qid,
                "question": str(gen.get("question") or ""),
                "category": str((p6 or {}).get("category") or p5_meta.get("question_type") or ""),
                "difficulty": str(p5_meta.get("difficulty") or ""),
                "context": ctx,
                "has_context": len(ctx.strip()) > 0,
                "reference_answer": str((p5 or {}).get("reference_answer") or (p6 or {}).get("reference_answer") or ""),
                "expected_behavior": str((p6 or {}).get("expected_behavior") or p5_meta.get("expected_behavior") or ""),
                "hallucination_test_type": str((p6 or {}).get("hallucination_test_type") or p5_meta.get("hallucination_test_type") or ""),
                "model_key": str(gen.get("model_key") or ""),
                "model_id": mid,
                "model_backend": str(gen.get("backend") or ""),
                "generated_answer": str(gen.get("generated_answer") or ""),
                "generation_status": str(gen.get("status") or "unknown"),
                "generation_time_seconds": gen.get("generation_time_seconds"),
                "generation_error": gen.get("error_message"),
                "exact_match": p5.get("exact_match") if p5 else None,
                "normalized_exact_match": p5.get("normalized_exact_match") if p5 else None,
                "token_precision": p5.get("token_precision") if p5 else None,
                "token_recall": p5.get("token_recall") if p5 else None,
                "token_f1": p5.get("token_f1") if p5 else None,
                "semantic_similarity": p5.get("semantic_similarity") if p5 else None,
                "correctness_score": p5_corr.get("correctness_score") if p5 else None,
                "correctness_label": p5_corr.get("correctness_label") if p5 else None,
                "key_info_coverage": p5_corr.get("key_info_coverage") if p5 else None,
                "has_numeric_contradiction": p5_corr.get("has_numeric_contradiction") if p5 else None,
                "reference_evaluation_status": p5.get("evaluation_status") if p5 else "pending",
                "is_counterfactual_context": bool(p6.get("is_counterfactual_context")) if p6 else False,
                "supported_claims": p6_faith.get("supported_count") if p6 else None,
                "contradicted_claims": p6_faith.get("contradicted_count") if p6 else None,
                "unsupported_claims": p6_faith.get("unsupported_count") if p6 else None,
                "total_factual_claims": p6_faith.get("total_factual_claims") if p6 else None,
                "faithfulness_ratio": p6_faith.get("faithfulness_ratio") if p6 else None,
                "contradiction_rate": p6_faith.get("contradiction_rate") if p6 else None,
                "unsupported_claim_rate": p6_faith.get("unsupported_claim_rate") if p6 else None,
                "context_hallucination_rate": p6.get("context_hallucination_rate") if p6 else None,
                "hallucination_detected": p6.get("hallucination_detected") if p6 else None,
                "hallucination_types": p6.get("hallucination_types") or [] if p6 else [],
                "hallucination_detection_status": p6.get("detection_status") if p6 else "pending",
                "hallucination_explanation": p6.get("explanation") if p6 else None,
                "claims": p6.get("claims") or [] if p6 else [],
                "judge_model_id": p7.get("judge_model_id") if p7 else None,
                "judge_prompt_version": p7.get("prompt_version") if p7 else None,
                "judge_relevance_score": p7.get("relevance_score") if p7 else None,
                "judge_factual_correctness_score": p7.get("factual_correctness_score") if p7 else None,
                "judge_completeness_score": p7.get("completeness_score") if p7 else None,
                "judge_contextual_faithfulness_score": p7.get("contextual_faithfulness_score") if p7 else None,
                "judge_clarity_score": p7.get("clarity_score") if p7 else None,
                "judge_overall_score": p7.get("overall_score") if p7 else None,
                "judge_confidence": p7.get("judge_confidence") if p7 else None,
                "judge_evaluation_status": p7.get("evaluation_status") if p7 else "pending",
                "judge_latency_seconds": p7.get("judge_latency_seconds") if p7 else None,
                "judge_error_message": p7.get("error_message") if p7 else None,
                "judge_dimension_explanations": p7.get("dimension_explanations") or {} if p7 else {},
                "raw_judge_response": p7.get("raw_judge_response") if p7 else None,
            }
        )
    return records


def load_run_data(run_dir: Path) -> LoadedRunData:
    """
    Load all available artifacts for a run directory, supporting complete,
    incomplete, and partially corrupted runs without crashing.
    """
    run_path = Path(run_dir)
    warnings: List[str] = []

    metadata_raw = _safe_read_json(run_path / "run_metadata.json", warnings, "run_metadata.json")
    run_cfg_raw = _safe_read_json(run_path / "run_config.json", warnings, "run_config.json")
    combined_raw = _safe_read_json(run_path / "combined_results.json", warnings, "combined_results.json")
    summary_raw = _safe_read_json(run_path / "model_summary.json", warnings, "model_summary.json")

    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    run_cfg = run_cfg_raw if isinstance(run_cfg_raw, dict) else {}

    combined_results: List[Dict[str, Any]]
    if isinstance(combined_raw, list):
        combined_results = [item for item in combined_raw if isinstance(item, dict)]
    elif (run_path / "question_level_results.csv").exists():
        try:
            csv_df = pd.read_csv(run_path / "question_level_results.csv", keep_default_na=False)
            combined_results = csv_df.to_dict(orient="records")
            warnings.append("Loaded records from question_level_results.csv fallback.")
        except Exception as exc:
            warnings.append(f"Failed reading question_level_results.csv: {exc}")
            combined_results = []
    else:
        combined_results = _reconstruct_partial_records(run_path, warnings)
        if combined_results:
            warnings.append("Reconstructed partial view from checkpoint files (run incomplete).")

    if isinstance(summary_raw, dict) and "models" in summary_raw:
        summary = summary_raw
    else:
        summary = aggregate_pipeline_results(combined_results)

    if warnings and not combined_results:
        status = "corrupted"
    elif metadata and isinstance(combined_raw, list):
        status = "complete"
    else:
        status = "incomplete"

    started_at = str(metadata.get("started_at_utc") or "")
    if not started_at and run_path.exists():
        started_at = pd.Timestamp(run_path.stat().st_mtime, unit="s", tz="UTC").isoformat()

    models = _extract_model_ids_from_config(run_cfg, metadata)
    if not models and combined_results:
        models = sorted({str(r.get("model_id", "")) for r in combined_results if r.get("model_id")})

    descriptor = RunDescriptor(
        run_id=run_path.name,
        run_dir=run_path,
        started_at_utc=started_at,
        finished_at_utc=str(metadata.get("finished_at_utc") or ""),
        status=status,
        models=models,
        dataset_hash=str(metadata.get("dataset_hash") or ""),
        total_pairs=len(combined_results),
        judge_enabled=bool(
            (metadata.get("enabled_stages") or {}).get("llm_judge")
            or (run_cfg.get("config") or {}).get("enable_judge")
            or any(
                r.get("judge_evaluation_status") not in {None, "disabled", "pending"}
                for r in combined_results
            )
        ),
        warnings=warnings,
    )

    return LoadedRunData(
        descriptor=descriptor,
        combined_results=combined_results,
        summary=summary,
        metadata=metadata,
        run_config=run_cfg,
        warnings=warnings,
    )


def filter_records(
    records: Sequence[Mapping[str, Any]],
    models: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    difficulties: Optional[Sequence[str]] = None,
    question_ids: Optional[Sequence[str]] = None,
    hallucination_test_types: Optional[Sequence[str]] = None,
    generation_statuses: Optional[Sequence[str]] = None,
    correctness_labels: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Filter combined question-model records dynamically across dashboard selectors."""
    model_set = set(models) if models else None
    cat_set = set(categories) if categories else None
    diff_set = {d.lower() for d in difficulties} if difficulties else None
    qid_set = set(question_ids) if question_ids else None
    h_type_set = set(hallucination_test_types) if hallucination_test_types else None
    gen_set = set(generation_statuses) if generation_statuses else None
    corr_set = set(correctness_labels) if correctness_labels else None

    filtered: List[Dict[str, Any]] = []
    for rec in records:
        if model_set is not None and str(rec.get("model_id", "")) not in model_set:
            continue
        if cat_set is not None and str(rec.get("category", "")) not in cat_set:
            continue
        if diff_set is not None and str(rec.get("difficulty", "")).lower() not in diff_set:
            continue
        if qid_set is not None and str(rec.get("question_id", "")) not in qid_set:
            continue
        if h_type_set is not None and str(rec.get("hallucination_test_type", "")) not in h_type_set:
            continue
        if gen_set is not None and str(rec.get("generation_status", "")) not in gen_set:
            continue
        if corr_set is not None and str(rec.get("correctness_label") or "") not in corr_set:
            continue
        filtered.append(dict(rec))

    return filtered


def records_to_export_csv_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize filtered question-model records to UTF-8 CSV bytes."""
    if not records:
        return b""
    df = combined_results_to_dataframe([dict(r) for r in records])
    return df.to_csv(index=False).encode("utf-8")


def run_payload_to_export_json_bytes(
    run_data: LoadedRunData,
    filtered_records: Sequence[Mapping[str, Any]],
    filtered_summary: Mapping[str, Any],
) -> bytes:
    """Serialize run metadata, summary, and filtered records into JSON bytes."""
    payload = {
        "run_id": run_data.descriptor.run_id,
        "status": run_data.descriptor.status,
        "metadata": run_data.metadata,
        "summary": dict(filtered_summary),
        "records": [dict(r) for r in filtered_records],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
