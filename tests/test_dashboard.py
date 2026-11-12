"""Unit and UI-smoke tests for the Phase 9 Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard.dashboard_utils import (
    build_claim_distribution_chart,
    build_judge_dimensions_chart,
    build_model_comparison_bar_chart,
    build_model_radar_chart,
    format_metric,
    format_metric_with_n,
    is_env_key_configured,
)
from src.dashboard.data_loader import (
    discover_runs,
    filter_records,
    load_run_data,
    records_to_export_csv_bytes,
    run_payload_to_export_json_bytes,
)
from src.pipeline.aggregation import aggregate_pipeline_results


def _sample_combined_records() -> List[Dict[str, Any]]:
    return [
        {
            "question_id": "Q001",
            "question": "What is the capital city of Australia?",
            "category": "General Knowledge",
            "difficulty": "easy",
            "context": "",
            "has_context": False,
            "reference_answer": "Canberra is the capital city of Australia.",
            "expected_behavior": "factual_answer",
            "hallucination_test_type": "factual_precision",
            "model_key": "qwen2.5-0.5b",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_backend": "huggingface_local",
            "generated_answer": "Canberra.",
            "generation_status": "success",
            "generation_time_seconds": 0.42,
            "generation_error": None,
            "exact_match": 0.0,
            "normalized_exact_match": 0.0,
            "token_precision": 1.0,
            "token_recall": 0.25,
            "token_f1": 0.4,
            "semantic_similarity": 0.85,
            "correctness_score": 0.72,
            "correctness_label": "correct",
            "key_info_coverage": 0.5,
            "has_numeric_contradiction": False,
            "reference_evaluation_status": "evaluated",
            "is_counterfactual_context": False,
            "supported_claims": 0,
            "contradicted_claims": 0,
            "unsupported_claims": 0,
            "total_factual_claims": 1,
            "faithfulness_ratio": None,
            "contradiction_rate": None,
            "unsupported_claim_rate": None,
            "context_hallucination_rate": None,
            "hallucination_detected": False,
            "hallucination_types": [],
            "hallucination_detection_status": "no_context_evaluated",
            "hallucination_explanation": "No context supplied; context-faithfulness skipped.",
            "claims": [],
            "judge_model_id": None,
            "judge_prompt_version": None,
            "judge_relevance_score": None,
            "judge_factual_correctness_score": None,
            "judge_completeness_score": None,
            "judge_contextual_faithfulness_score": None,
            "judge_clarity_score": None,
            "judge_overall_score": None,
            "judge_confidence": None,
            "judge_evaluation_status": "disabled",
            "judge_latency_seconds": None,
            "judge_error_message": None,
            "judge_dimension_explanations": {},
            "raw_judge_response": None,
        },
        {
            "question_id": "Q066",
            "question": "How many satellites in Phase 2?",
            "category": "Context-based Question Answering",
            "difficulty": "easy",
            "context": "Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
            "has_context": True,
            "reference_answer": "18 synthetic-aperture radar satellites.",
            "expected_behavior": "context_grounded_answer",
            "hallucination_test_type": "context_supported_precision",
            "model_key": "qwen2.5-0.5b",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_backend": "huggingface_local",
            "generated_answer": "Phase 2 added 24 radar satellites.",
            "generation_status": "success",
            "generation_time_seconds": 0.55,
            "generation_error": None,
            "exact_match": 0.0,
            "normalized_exact_match": 0.0,
            "token_precision": 0.6,
            "token_recall": 0.75,
            "token_f1": 0.6667,
            "semantic_similarity": 0.89,
            "correctness_score": 0.3,
            "correctness_label": "contradictory_numeric",
            "key_info_coverage": 0.5,
            "has_numeric_contradiction": True,
            "reference_evaluation_status": "evaluated",
            "is_counterfactual_context": False,
            "supported_claims": 0,
            "contradicted_claims": 1,
            "unsupported_claims": 0,
            "total_factual_claims": 1,
            "faithfulness_ratio": 0.0,
            "contradiction_rate": 1.0,
            "unsupported_claim_rate": 0.0,
            "context_hallucination_rate": 1.0,
            "hallucination_detected": True,
            "hallucination_types": ["context_contradiction"],
            "hallucination_detection_status": "computed",
            "hallucination_explanation": "1 claim(s) contradicted supplied context.",
            "claims": [
                {
                    "claim_id": "Q066_C01",
                    "question_id": "Q066",
                    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "claim_text": "Phase 2 added 24 radar satellites",
                    "claim_type": "numeric_claim",
                    "retrieved_evidence": ["Phase 2 added 18 synthetic-aperture radar satellites in 2023."],
                    "evidence_scores": [0.91],
                    "verification_label": "CONTRADICTED",
                    "confidence": 0.92,
                    "evidence_strength": 0.91,
                    "explanation": "Numeric conflict: 24 vs 18.",
                    "verifier_backend": "hybrid_semantic_rule",
                    "details": {},
                }
            ],
            "judge_model_id": None,
            "judge_prompt_version": None,
            "judge_relevance_score": None,
            "judge_factual_correctness_score": None,
            "judge_completeness_score": None,
            "judge_contextual_faithfulness_score": None,
            "judge_clarity_score": None,
            "judge_overall_score": None,
            "judge_confidence": None,
            "judge_evaluation_status": "disabled",
            "judge_latency_seconds": None,
            "judge_error_message": None,
            "judge_dimension_explanations": {},
            "raw_judge_response": None,
        },
    ]


def _create_complete_run(runs_dir: Path, run_id: str) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records = _sample_combined_records()
    summary = aggregate_pipeline_results(records)
    metadata = {
        "run_id": run_id,
        "started_at_utc": "2026-11-12T10:00:00+00:00",
        "finished_at_utc": "2026-11-12T10:01:00+00:00",
        "dataset_hash": "abc123def456",
        "models": [{"key": "qwen2.5-0.5b", "model_id": "Qwen/Qwen2.5-0.5B-Instruct"}],
        "enabled_stages": {"reference_evaluation": True, "hallucination_detection": True, "llm_judge": False},
        "omitted_stage_files": {"judge_results.json": "Disabled"},
        "stage_counts": {"total_question_model_pairs": len(records)},
    }
    (run_dir / "combined_results.json").write_text(json.dumps(records), encoding="utf-8")
    (run_dir / "model_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps({"config": {}, "fingerprint": {}}), encoding="utf-8")
    return run_dir


def test_discover_and_load_complete_incomplete_and_corrupted_runs(tmp_path) -> None:
    """Verify run discovery handles empty dirs, complete runs, partial checkpoints, and corrupt JSON."""
    runs_dir = tmp_path / "runs"
    assert discover_runs(runs_dir) == []

    complete_dir = _create_complete_run(runs_dir, "run_complete")

    # Incomplete run with only generation_results.json
    incomplete_dir = runs_dir / "run_incomplete"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "generation_results.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "Q001",
                    "model_key": "qwen2.5-0.5b",
                    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "backend": "huggingface_local",
                    "question": "Capital of Australia?",
                    "context": "",
                    "generated_answer": "Canberra.",
                    "status": "success",
                    "generation_time_seconds": 0.3,
                }
            ]
        ),
        encoding="utf-8",
    )

    # Corrupted run
    corrupt_dir = runs_dir / "run_corrupt"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "run_metadata.json").write_text("{invalid json", encoding="utf-8")

    discovered = discover_runs(runs_dir)
    statuses = {d.run_id: d.status for d in discovered}
    assert statuses["run_complete"] == "complete"
    assert statuses["run_incomplete"] == "incomplete"
    assert statuses["run_corrupt"] == "corrupted"

    loaded_complete = load_run_data(complete_dir)
    assert loaded_complete.descriptor.status == "complete"
    assert len(loaded_complete.combined_results) == 2

    loaded_incomplete = load_run_data(incomplete_dir)
    assert loaded_incomplete.descriptor.status == "incomplete"
    assert len(loaded_incomplete.combined_results) == 1

    loaded_corrupt = load_run_data(corrupt_dir)
    assert loaded_corrupt.descriptor.status == "corrupted"
    assert len(loaded_corrupt.warnings) >= 1


def test_filtering_exports_and_chart_missing_value_preservation(tmp_path) -> None:
    """Verify multi-filter logic, CSV/JSON export bytes, and Plotly missing-metric preservation."""
    run_dir = _create_complete_run(tmp_path / "runs", "run_complete")
    loaded = load_run_data(run_dir)

    filtered = filter_records(
        loaded.combined_results,
        categories=["Context-based Question Answering"],
        correctness_labels=["contradictory_numeric"],
    )
    assert len(filtered) == 1
    assert filtered[0]["question_id"] == "Q066"

    csv_bytes = records_to_export_csv_bytes(filtered)
    assert b"Q066" in csv_bytes
    json_bytes = run_payload_to_export_json_bytes(loaded, filtered, aggregate_pipeline_results(filtered))
    parsed_export = json.loads(json_bytes.decode("utf-8"))
    assert parsed_export["run_id"] == "run_complete"
    assert len(parsed_export["records"]) == 1

    # Missing metrics must stay None (never replaced with 0) in bar charts & formatters
    assert format_metric(None) == "N/A"
    assert format_metric_with_n({"mean": None, "n": 0}) == "N/A (n=0)"
    fig = build_model_comparison_bar_chart(
        summary_models=loaded.summary["models"],
        metric_specs=[
            ("reference_evaluation", "semantic_similarity", "Semantic Sim."),
            ("llm_judge", "overall_score", "Judge Overall"),
        ],
        title="Test Missing Preservation",
    )
    y_vals = list(fig.data[0].y)
    assert y_vals[0] == pytest.approx(0.87)
    assert y_vals[1] is None  # Judge disabled -> None, NOT 0!

    radar_fig = build_model_radar_chart(loaded.summary["models"])
    assert len(radar_fig.data) == 1

    claim_fig = build_claim_distribution_chart(loaded.combined_results)
    assert len(claim_fig.data) == 3

    judge_fig = build_judge_dimensions_chart(loaded.summary["models"])
    assert len(judge_fig.data) == 1


def test_env_key_check_never_exposes_secret_value(monkeypatch) -> None:
    """Verify credential checker returns only boolean availability."""
    monkeypatch.setenv("TEST_SECRET_KEY", "super-secret-value")
    assert is_env_key_configured("TEST_SECRET_KEY") is True
    assert is_env_key_configured("MISSING_SECRET_KEY_XYZ") is False


APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def test_streamlit_app_renders_empty_state_without_crashing(monkeypatch, tmp_path) -> None:
    """Smoke-test app.py via Streamlit AppTest when no runs exist."""
    class _EmptyPaths:
        results_dir = tmp_path
        default_dataset_path = Path(__file__).resolve().parents[1] / "data" / "evaluation_dataset.csv"
        processed_data_dir = tmp_path / "processed"

        def ensure_directories(self) -> None:
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def _fake_get_config(ensure_dirs=True):
        return type("Cfg", (), {"paths": _EmptyPaths()})()

    monkeypatch.setattr("src.utils.config.get_config", _fake_get_config)
    monkeypatch.setattr("src.dashboard.overview.get_config", _fake_get_config, raising=False)
    monkeypatch.setattr("src.dashboard.experiment_management.get_config", _fake_get_config, raising=False)
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception
    assert any("Evaluation Overview" in h.value for h in at.header)


def test_streamlit_app_renders_all_pages_with_populated_run(monkeypatch, tmp_path) -> None:
    """Smoke-test all 6 pages in app.py against a populated synthetic run."""
    runs_dir = tmp_path / "runs"
    _create_complete_run(runs_dir, "run_populated")

    class _Paths:
        results_dir = tmp_path
        default_dataset_path = Path(__file__).resolve().parents[1] / "data" / "evaluation_dataset.csv"
        processed_data_dir = tmp_path / "processed"

        def ensure_directories(self) -> None:
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def _fake_get_config(ensure_dirs=True):
        return type("Cfg", (), {"paths": _Paths()})()

    monkeypatch.setattr("src.utils.config.get_config", _fake_get_config)
    monkeypatch.setattr("src.dashboard.overview.get_config", _fake_get_config, raising=False)
    monkeypatch.setattr("src.dashboard.experiment_management.get_config", _fake_get_config, raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run()
    assert not at.exception

    expected_headers = {
        "Overview": "Evaluation Overview",
        "Model Comparison": "Model Comparison",
        "Hallucination Analysis": "Hallucination & Contextual Faithfulness Analysis",
        "Question-Level Analysis": "Question-Level Analysis",
        "LLM-as-a-Judge": "LLM-as-a-Judge Evaluation",
        "Experiment Management": "Experiment Management",
    }
    for page_name, header_text in expected_headers.items():
        at.sidebar.radio[0].set_value(page_name).run()
        assert not at.exception, f"Page '{page_name}' raised exception: {at.exception}"
        assert any(header_text in h.value for h in at.header), f"Missing header on '{page_name}'"


def test_response_time_denominator_and_correctness_threshold_consistency() -> None:
    """Verify response-time excludes error/missing/NaN while including success + empty_response."""
    from src.evaluation.correctness import CorrectnessConfig

    cfg = CorrectnessConfig()
    assert cfg.correct_threshold == pytest.approx(0.72)
    assert cfg.partial_threshold == pytest.approx(0.42)

    records = [
        {"model_id": "m1", "category": "factual", "generation_status": "success", "generation_time_seconds": 2.0},
        {"model_id": "m1", "category": "factual", "generation_status": "empty_response", "generation_time_seconds": 4.0},
        {"model_id": "m1", "category": "factual", "generation_status": "error", "generation_time_seconds": 99.0},
        {"model_id": "m1", "category": "factual", "generation_status": "missing", "generation_time_seconds": 50.0},
        {"model_id": "m1", "category": "factual", "generation_status": "success", "generation_time_seconds": None},
        {"model_id": "m1", "category": "factual", "generation_status": "success", "generation_time_seconds": float("nan")},
    ]
    agg = aggregate_pipeline_results(records)
    gen = agg["models"]["m1"]["generation"]
    assert gen["successful_count"] == 3
    assert gen["empty_response_count"] == 1
    assert gen["failed_count"] == 1
    assert gen["response_time_seconds"]["n"] == 2
    assert gen["response_time_seconds"]["mean"] == pytest.approx(3.0)

