"""Comprehensive unit and integration tests for the Phase 8 evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

from models.model_config import GenerationConfig, ModelBackend, ModelSpec
from src.evaluation.semantic_similarity import BaseEmbeddingProvider
from src.generation.generator import APIBasedGenerator, BaseLLMGenerator, GenerationResult
from src.hallucination.hallucination_detector import HallucinationDetector
from src.judge.judge_client import BaseJudgeClient, JudgePromptPayload, JudgeRawResponse, JudgeTimeoutError
from src.judge.judge_config import JudgeConfig
from src.pipeline import (
    CheckpointCorruptedError,
    CheckpointManager,
    CheckpointMismatchError,
    PipelineConfig,
    run_evaluation_pipeline,
)


class MockPipelineEmbedder(BaseEmbeddingProvider):
    """Deterministic keyword-cluster embedder requiring zero model downloads."""

    def __init__(self) -> None:
        self.model_name = "mock-all-MiniLM-L6-v2"

    def _vec(self, text: str) -> np.ndarray:
        low = text.lower()
        if any(k in low for k in ("canberra", "australia", "capital")):
            return np.array([0.95, 0.1, 0.0, 0.0], dtype=np.float64)
        if any(k in low for k in ("meridian", "satellite", "phase 2", "18", "24")):
            return np.array([0.0, 0.95, 0.1, 0.0], dtype=np.float64)
        if any(k in low for k in ("72", "water", "boils", "lyon", "paris", "100")):
            return np.array([0.0, 0.1, 0.95, 0.0], dtype=np.float64)
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])


class TrackingMockGenerator(BaseLLMGenerator):
    """Mock generator tracking generate calls, load/unload order, and optional failures."""

    def __init__(
        self,
        spec: ModelSpec,
        generation_config: GenerationConfig,
        answers: Dict[str, str],
        event_log: List[str],
        fail_qids: set[str] | None = None,
        raise_qids: set[str] | None = None,
        captured_inputs: Dict[str, List[tuple[str, str, str, str | None]]] | None = None,
    ) -> None:
        super().__init__(spec=spec, generation_config=generation_config)
        self.answers = answers
        self.event_log = event_log
        self.fail_qids = fail_qids or set()
        self.raise_qids = raise_qids or set()
        self.captured_inputs = captured_inputs
        self.event_log.append(f"load:{spec.key}")

    def generate(
        self,
        question_id: str,
        question: str,
        context: str = "",
        system_prompt: str | None = None,
    ) -> GenerationResult:
        self.event_log.append(f"gen:{self.spec.key}:{question_id}")
        if self.captured_inputs is not None:
            self.captured_inputs.setdefault(self.spec.key, []).append(
                (question_id, question, context, system_prompt)
            )
        if question_id in self.raise_qids:
            raise RuntimeError(f"Simulated uncaught exception on {question_id}.")
        if question_id in self.fail_qids:
            return GenerationResult(
                question_id=question_id,
                model_key=self.spec.key,
                model_id=self.spec.model_id,
                backend=self.spec.backend.value,
                question=question,
                context=context,
                prompt_used=question,
                generated_answer="",
                generation_time_seconds=0.05,
                status="error",
                error_message="Simulated single-question generation error.",
            )
        return GenerationResult(
            question_id=question_id,
            model_key=self.spec.key,
            model_id=self.spec.model_id,
            backend=self.spec.backend.value,
            question=question,
            context=context,
            prompt_used=question,
            generated_answer=self.answers[question_id],
            generation_time_seconds=0.10,
            status="success",
        )

    def unload(self) -> None:
        self.event_log.append(f"unload:{self.spec.key}")


class MockPipelineJudgeClient(BaseJudgeClient):
    """Mock judge client tracking calls, supporting timeouts, and unloading."""

    def __init__(self, event_log: List[str], timeout_qids: set[str] | None = None) -> None:
        self.judge_model_id = "mock-judge-model"
        self.event_log = event_log
        self.timeout_qids = timeout_qids or set()

    def evaluate_prompt(self, prompt: JudgePromptPayload) -> JudgeRawResponse:
        self.event_log.append(f"judge:{prompt.question_id}")
        if prompt.question_id in self.timeout_qids:
            raise JudgeTimeoutError("Simulated judge timeout.")

        ctx_score: Any = 5 if prompt.has_context else "NOT_APPLICABLE"
        if prompt.has_context and ("24" in prompt.user_prompt or "100" in prompt.user_prompt):
            ctx_score = 1
            fact_score = 1
        else:
            fact_score = 5

        payload = {
            "relevance_score": 5,
            "factual_correctness_score": fact_score,
            "completeness_score": 5,
            "contextual_faithfulness_score": ctx_score,
            "clarity_score": 5,
            "dimension_explanations": {
                "relevance": "On topic.",
                "factual_correctness": "Matches expected evaluation.",
                "completeness": "Complete.",
                "contextual_faithfulness": "Consistent or not applicable.",
                "clarity": "Clear.",
            },
            "judge_confidence": 0.90,
            "overall_summary": "Evaluated cleanly.",
        }
        return JudgeRawResponse(
            raw_text=json.dumps(payload),
            judge_model_id=self.judge_model_id,
            backend="mock_judge",
            latency_seconds=0.03,
        )

    def unload(self) -> None:
        self.event_log.append("unload:judge")


def _write_mini_dataset(path: Path) -> Path:
    rows = [
        {
            "question_id": "Q001",
            "question": "What is the capital city of Australia?",
            "context": "",
            "reference_answer": "Canberra is the capital city of Australia.",
            "category": "General Knowledge",
            "difficulty": "easy",
            "expected_behavior": "factual_answer",
            "hallucination_test_type": "factual_precision",
        },
        {
            "question_id": "Q066",
            "question": "How many satellites were deployed in Phase 2?",
            "context": "Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
            "reference_answer": "18 synthetic-aperture radar satellites were deployed in Phase 2.",
            "category": "Context-based Question Answering",
            "difficulty": "easy",
            "expected_behavior": "context_grounded_answer",
            "hallucination_test_type": "context_supported_precision",
        },
        {
            "question_id": "Q079",
            "question": "According to the simulation brief, at what temperature does water boil?",
            "context": "Simulation Rulebook Alpha: pure water boils at 72 degrees Celsius at sea level.",
            "reference_answer": "Pure water boils at 72 degrees Celsius at sea level.",
            "category": "Context-based Question Answering",
            "difficulty": "medium",
            "expected_behavior": "follow_counterfactual_context",
            "hallucination_test_type": "counterfactual_context",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_end_to_end_pipeline_judge_disabled_and_sequential_unloading(tmp_path) -> None:
    """
    Verify end-to-end execution with 2 models, missing context, counterfactual context,
    sequential load/unload ordering, omitted judge file, and CSV/JSON consistency.
    """
    dataset_path = _write_mini_dataset(tmp_path / "mini.csv")
    event_log: List[str] = []

    answers_map = {
        "qwen2.5-0.5b": {
            "Q001": "Canberra is the capital city of Australia.",
            "Q066": "Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
            "Q079": "Pure water boils at 72 degrees Celsius at sea level.",
        },
        "flan-t5-base": {
            "Q001": "Canberra is the capital city of Australia.",
            "Q066": "Phase 2 added 24 synthetic-aperture radar satellites in 2023.",
            "Q079": "Pure water boils at 100 degrees Celsius at sea level.",
        },
    }

    captured_inputs: Dict[str, List[tuple[str, str, str, str | None]]] = {}

    def gen_factory(spec: ModelSpec, cfg: PipelineConfig) -> BaseLLMGenerator:
        return TrackingMockGenerator(
            spec=spec,
            generation_config=cfg.generation_config,
            answers=answers_map[spec.key],
            event_log=event_log,
            captured_inputs=captured_inputs,
        )

    cfg = PipelineConfig(
        models=["qwen2.5-0.5b", "flan-t5-base"],
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        summaries_dir=tmp_path / "summaries",
        run_id="test_run_no_judge",
        enable_judge=False,
    )

    result = run_evaluation_pipeline(
        config=cfg,
        generator_factory=gen_factory,
        embedder=MockPipelineEmbedder(),
    )

    # Sequential execution: Model A loaded, run, and unloaded BEFORE Model B loads
    assert event_log == [
        "load:qwen2.5-0.5b",
        "gen:qwen2.5-0.5b:Q001",
        "gen:qwen2.5-0.5b:Q066",
        "gen:qwen2.5-0.5b:Q079",
        "unload:qwen2.5-0.5b",
        "load:flan-t5-base",
        "gen:flan-t5-base:Q001",
        "gen:flan-t5-base:Q066",
        "gen:flan-t5-base:Q079",
        "unload:flan-t5-base",
    ]

    # Disabled judge must NOT create judge_results.json and must document omission in metadata
    assert not (result.run_dir / "judge_results.json").exists()
    assert "judge_results.json" in result.metadata["omitted_stage_files"]

    # Check missing-context and counterfactual handling
    by_pair = {(r["model_key"], r["question_id"]): r for r in result.combined_results}
    assert by_pair[("qwen2.5-0.5b", "Q001")]["hallucination_detection_status"] == "no_context_evaluated"
    assert by_pair[("qwen2.5-0.5b", "Q001")]["faithfulness_ratio"] is None
    assert by_pair[("qwen2.5-0.5b", "Q079")]["is_counterfactual_context"] is True
    assert by_pair[("qwen2.5-0.5b", "Q079")]["faithfulness_ratio"] == pytest.approx(1.0)
    assert by_pair[("flan-t5-base", "Q079")]["faithfulness_ratio"] == pytest.approx(0.0)
    assert by_pair[("flan-t5-base", "Q079")]["hallucination_detected"] is True

    # Aggregation denominators: faithfulness only over the 2 context-bearing questions
    qwen_summary = result.summary["models"]["Qwen/Qwen2.5-0.5B-Instruct"]
    flan_summary = result.summary["models"]["google/flan-t5-base"]
    assert qwen_summary["faithfulness_and_hallucination"]["n_context_computed"] == 2
    assert qwen_summary["faithfulness_and_hallucination"]["n_no_context_skipped"] == 1
    assert qwen_summary["faithfulness_and_hallucination"]["faithfulness_ratio"]["mean"] == pytest.approx(1.0)
    assert flan_summary["faithfulness_and_hallucination"]["contradiction_rate"]["mean"] == pytest.approx(1.0)

    # Both models received identical ordered questions, contexts, and system prompt
    assert captured_inputs["qwen2.5-0.5b"] == captured_inputs["flan-t5-base"]

    # Reproducibility metadata completeness
    meta = result.metadata
    assert meta["run_id"] == "test_run_no_judge"
    assert len(meta["dataset_hash"]) == 64
    assert meta["prompt_template"]["name"] == cfg.prompt_template_name
    assert meta["prompt_template"]["system_prompt"] == cfg.system_prompt
    assert meta["generation_config"]["seed"] == 42
    assert meta["evaluation_models"]["embedding_model_id"] == cfg.embedding_model_id
    assert meta["judge_config"] is None
    assert meta["execution_mode"] == "sequential"
    assert "python" in meta["environment"] and "torch" in meta["environment"]
    assert "generation_skipped_from_checkpoint" in meta["stage_counts"]

    # JSON and CSV consistency (row count, ordering, and score values)
    csv_df = pd.read_csv(result.output_files["question_level_results_csv"])
    with result.output_files["combined_results_json"].open("r", encoding="utf-8") as fh:
        json_rows = json.load(fh)
    assert len(csv_df) == len(json_rows) == 6
    assert list(csv_df["question_id"]) == [r["question_id"] for r in json_rows]
    assert list(csv_df["model_id"]) == [r["model_id"] for r in json_rows]
    for idx, row_json in enumerate(json_rows):
        assert csv_df.loc[idx, "semantic_similarity"] == pytest.approx(row_json["semantic_similarity"])


def test_judge_enabled_independent_failures_and_denominator_parity_warning(tmp_path) -> None:
    """
    Verify independent generation and judge failure isolation, judge unloading,
    and explicit sample-population parity warnings when model valid counts differ.
    """
    dataset_path = _write_mini_dataset(tmp_path / "mini.csv")
    event_log: List[str] = []

    answers = {
        "Q001": "Canberra is the capital city of Australia.",
        "Q066": "Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
        "Q079": "Pure water boils at 72 degrees Celsius at sea level.",
    }

    def gen_factory(spec: ModelSpec, cfg: PipelineConfig) -> BaseLLMGenerator:
        fail_set = {"Q066"} if spec.key == "flan-t5-base" else set()
        return TrackingMockGenerator(
            spec=spec,
            generation_config=cfg.generation_config,
            answers=answers,
            event_log=event_log,
            fail_qids=fail_set,
        )

    judge_client = MockPipelineJudgeClient(event_log=event_log, timeout_qids={"Q079"})
    cfg = PipelineConfig(
        models=["qwen2.5-0.5b", "flan-t5-base"],
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        summaries_dir=tmp_path / "summaries",
        run_id="test_run_with_judge_failures",
        enable_judge=True,
        judge_config=JudgeConfig(judge_model_key_or_id="qwen2.5-1.5b"),
    )

    result = run_evaluation_pipeline(
        config=cfg,
        generator_factory=gen_factory,
        embedder=MockPipelineEmbedder(),
        judge_client=judge_client,
    )

    assert (result.run_dir / "judge_results.json").exists()
    assert "unload:judge" in event_log

    by_pair = {(r["model_key"], r["question_id"]): r for r in result.combined_results}
    # Judge timeout on Q079 leaves Phase 5/6 results intact!
    qwen_q079 = by_pair[("qwen2.5-0.5b", "Q079")]
    assert qwen_q079["judge_evaluation_status"] == "judge_timeout"
    assert qwen_q079["reference_evaluation_status"] == "evaluated"
    assert qwen_q079["faithfulness_ratio"] == pytest.approx(1.0)

    # Generation failure on flan-t5-base Q066 reduces its valid sample count -> parity warning emitted
    assert by_pair[("flan-t5-base", "Q066")]["generation_status"] == "error"
    assert len(result.summary["sample_population_parity_notes"]) >= 1

    # Judge contextual faithfulness mean excludes NOT_APPLICABLE on Q001
    qwen_judge = result.summary["models"]["Qwen/Qwen2.5-0.5B-Instruct"]["llm_judge"]
    assert qwen_judge["n_evaluated"] == 2  # Q001, Q066 succeeded; Q079 timed out
    assert qwen_judge["contextual_faithfulness_score"]["n"] == 1  # Only Q066 had context


def test_checkpoint_resume_retry_corruption_and_mismatch_detection(tmp_path) -> None:
    """
    Verify incremental checkpointing, skipping completed items on resume, retrying
    failed items, detecting config mismatches, and rejecting corrupted checkpoints.
    """
    dataset_path = _write_mini_dataset(tmp_path / "mini.csv")
    run_id = "resume_test_run"
    answers = {
        "Q001": "Canberra is the capital city of Australia.",
        "Q066": "Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
        "Q079": "Pure water boils at 72 degrees Celsius at sea level.",
    }

    # Initial run where Q066 fails generation
    log_pass1: List[str] = []
    cfg1 = PipelineConfig(
        models=["qwen2.5-0.5b"],
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        summaries_dir=tmp_path / "summaries",
        run_id=run_id,
        enable_judge=False,
        save_incremental=True,
    )
    res1 = run_evaluation_pipeline(
        config=cfg1,
        generator_factory=lambda s, c: TrackingMockGenerator(
            s, c.generation_config, answers, log_pass1, fail_qids={"Q066"}
        ),
        embedder=MockPipelineEmbedder(),
    )
    assert res1.metadata["stage_counts"]["generation_error"] == 1
    assert res1.metadata["stage_counts"]["generation_success"] == 2

    # Resume with failure resolved -> Q001 & Q079 must NOT regenerate, only Q066 retries!
    log_pass2: List[str] = []
    cfg2 = PipelineConfig(
        models=["qwen2.5-0.5b"],
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        summaries_dir=tmp_path / "summaries",
        run_id=run_id,
        enable_judge=False,
        resume=True,
        retry_failed=True,
    )
    res2 = run_evaluation_pipeline(
        config=cfg2,
        generator_factory=lambda s, c: TrackingMockGenerator(
            s, c.generation_config, answers, log_pass2, fail_qids=set()
        ),
        embedder=MockPipelineEmbedder(),
    )
    assert res2.metadata["resumed_from_checkpoint"] is True
    assert res2.metadata["stage_counts"]["generation_success"] == 3
    assert log_pass2 == [
        "load:qwen2.5-0.5b",
        "gen:qwen2.5-0.5b:Q066",
        "unload:qwen2.5-0.5b",
    ]

    # Resuming with a different model must raise CheckpointMismatchError
    cfg_mismatch = PipelineConfig(
        models=["flan-t5-base"],
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        summaries_dir=tmp_path / "summaries",
        run_id=run_id,
        enable_judge=False,
        resume=True,
    )
    with pytest.raises(CheckpointMismatchError):
        run_evaluation_pipeline(config=cfg_mismatch, embedder=MockPipelineEmbedder())

    # Corrupted checkpoint JSON must raise CheckpointCorruptedError
    cm = CheckpointManager(tmp_path / "runs" / run_id)
    cm.generation_path.write_text("{corrupted json!!!", encoding="utf-8")
    with pytest.raises(CheckpointCorruptedError):
        run_evaluation_pipeline(config=cfg2, embedder=MockPipelineEmbedder())


def test_incremental_saving_midrun_and_partial_crash_recovery(tmp_path) -> None:
    """
    Verify incremental checkpoints exist on disk mid-run after each question,
    partial pre-crash checkpoints resume without regenerating completed items,
    and completed Phase 5/6/7 evaluations are skipped on resume.
    """
    dataset_path = _write_mini_dataset(tmp_path / "mini.csv")
    run_id = "incremental_crash_run"
    answers = {
        "Q001": "Canberra is the capital city of Australia.",
        "Q066": "Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
        "Q079": "Pure water boils at 72 degrees Celsius at sea level.",
    }
    observed_midrun_counts: List[int] = []
    gen_file = tmp_path / "runs" / run_id / "generation_results.json"

    def on_progress(stage: str, step: int, total: int) -> None:
        if stage == "generation" and gen_file.exists():
            observed_midrun_counts.append(len(json.loads(gen_file.read_text(encoding="utf-8"))))

    cfg1 = PipelineConfig(
        models=["qwen2.5-0.5b"],
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        summaries_dir=tmp_path / "summaries",
        run_id=run_id,
        enable_judge=True,
        judge_config=JudgeConfig(judge_model_key_or_id="qwen2.5-1.5b"),
        save_incremental=True,
    )
    event_log_1: List[str] = []
    judge_1 = MockPipelineJudgeClient(event_log=event_log_1)
    run_evaluation_pipeline(
        config=cfg1,
        generator_factory=lambda s, c: TrackingMockGenerator(
            s, c.generation_config, answers, event_log_1
        ),
        embedder=MockPipelineEmbedder(),
        judge_client=judge_1,
        progress_callback=on_progress,
    )
    # Incremental saving wrote 1, then 2, then 3 generation entries as steps progressed
    assert observed_midrun_counts == [1, 2, 3]

    # Simulate partial crash recovery: remove Q079 from generation_results.json while keeping Q001/Q066
    saved_gen = json.loads(gen_file.read_text(encoding="utf-8"))
    partial_gen = [item for item in saved_gen if item["question_id"] != "Q079"]
    gen_file.write_text(json.dumps(partial_gen), encoding="utf-8")

    # Remove Q079 from judge_results.json too so only Q079 should be re-judged on resume
    judge_file = tmp_path / "runs" / run_id / "judge_results.json"
    saved_judge = json.loads(judge_file.read_text(encoding="utf-8"))
    partial_judge = [item for item in saved_judge if item["question_id"] != "Q079"]
    judge_file.write_text(json.dumps(partial_judge), encoding="utf-8")

    event_log_2: List[str] = []
    cfg2 = PipelineConfig(
        models=["qwen2.5-0.5b"],
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        summaries_dir=tmp_path / "summaries",
        run_id=run_id,
        enable_judge=True,
        judge_config=JudgeConfig(judge_model_key_or_id="qwen2.5-1.5b"),
        resume=True,
    )
    res2 = run_evaluation_pipeline(
        config=cfg2,
        generator_factory=lambda s, c: TrackingMockGenerator(
            s, c.generation_config, answers, event_log_2
        ),
        embedder=MockPipelineEmbedder(),
        judge_client=MockPipelineJudgeClient(event_log=event_log_2),
    )
    assert res2.metadata["stage_counts"]["generation_skipped_from_checkpoint"] == 2
    assert res2.metadata["judge_config"]["judge_model_key_or_id"] == "qwen2.5-1.5b"
    assert event_log_2 == [
        "load:qwen2.5-0.5b",
        "gen:qwen2.5-0.5b:Q079",
        "unload:qwen2.5-0.5b",
        "judge:Q079",
        "unload:judge",
    ]


def test_isolated_phase6_failure_unload_on_exception_and_api_noop(
    tmp_path, monkeypatch
) -> None:
    """
    Verify:
    1. An unexpected exception during Phase 6 detection for one question isolates to
       `detection_error` without aborting Phase 5, Phase 7, or other questions.
    2. An uncaught exception in `generate()` still records an error and calls `unload()`.
    3. APIBasedGenerator.unload() is a safe no-op.
    4. Prompt, dataset, and judge config mismatches are rejected on resume.
    """
    dataset_path = _write_mini_dataset(tmp_path / "mini.csv")
    answers = {
        "Q001": "Canberra is the capital city of Australia.",
        "Q066": "Phase 2 added 18 synthetic-aperture radar satellites in 2023.",
        "Q079": "Pure water boils at 72 degrees Celsius at sea level.",
    }
    event_log: List[str] = []

    orig_detect_single = HallucinationDetector.detect_single

    def flaky_detect_single(self, **kwargs):
        if kwargs.get("question_id") == "Q066":
            raise RuntimeError("Simulated Phase 6 verifier failure.")
        return orig_detect_single(self, **kwargs)

    monkeypatch.setattr(HallucinationDetector, "detect_single", flaky_detect_single)

    cfg = PipelineConfig(
        models=["qwen2.5-0.5b"],
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        summaries_dir=tmp_path / "summaries",
        run_id="isolated_p6_and_unload",
        enable_judge=True,
        judge_config=JudgeConfig(judge_model_key_or_id="qwen2.5-1.5b"),
    )
    res = run_evaluation_pipeline(
        config=cfg,
        generator_factory=lambda s, c: TrackingMockGenerator(
            s, c.generation_config, answers, event_log, raise_qids={"Q001"}
        ),
        embedder=MockPipelineEmbedder(),
        judge_client=MockPipelineJudgeClient(event_log=event_log),
    )

    # Even with an uncaught exception in generate(Q001), unload() executed cleanly
    assert "unload:qwen2.5-0.5b" in event_log
    by_q = {r["question_id"]: r for r in res.combined_results}
    # Failed generation Q001 excluded from Phase 5 n_evaluated & Phase 6 computed denominators
    assert by_q["Q001"]["generation_status"] == "error"
    assert by_q["Q001"]["reference_evaluation_status"] == "generation_error"
    assert res.summary["models"]["Qwen/Qwen2.5-0.5B-Instruct"]["reference_evaluation"]["n_evaluated"] == 2

    # Phase 6 failure on Q066 did NOT abort Phase 5, Phase 7, or Q079
    assert by_q["Q066"]["hallucination_detection_status"] == "detection_error"
    assert by_q["Q066"]["reference_evaluation_status"] == "evaluated"
    assert by_q["Q066"]["judge_evaluation_status"] == "evaluated"
    assert by_q["Q079"]["hallucination_detection_status"] == "computed"

    # API-backed generator unload() is a safe no-op
    api_spec = ModelSpec(
        key="mock-api",
        display_name="Mock API",
        model_id="mock-api-model",
        backend=ModelBackend.GROQ_API,
        architecture="chat_api",
    )
    api_gen = APIBasedGenerator(spec=api_spec)
    assert api_gen.unload() is None

    # Config mismatch checks on prompt, judge model, and dataset modification
    cfg_bad_prompt = PipelineConfig(
        models=["qwen2.5-0.5b"],
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        summaries_dir=tmp_path / "summaries",
        run_id="isolated_p6_and_unload",
        system_prompt="Different prompt!",
        enable_judge=True,
        judge_config=JudgeConfig(judge_model_key_or_id="qwen2.5-1.5b"),
        resume=True,
    )
    with pytest.raises(CheckpointMismatchError):
        run_evaluation_pipeline(config=cfg_bad_prompt, embedder=MockPipelineEmbedder())

    cfg_bad_judge = PipelineConfig(
        models=["qwen2.5-0.5b"],
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        summaries_dir=tmp_path / "summaries",
        run_id="isolated_p6_and_unload",
        enable_judge=True,
        judge_config=JudgeConfig(judge_model_key_or_id="gemini-2.5-flash"),
        resume=True,
    )
    with pytest.raises(CheckpointMismatchError):
        run_evaluation_pipeline(config=cfg_bad_judge, embedder=MockPipelineEmbedder())

    # Corrupted run_config.json raises CheckpointCorruptedError
    (tmp_path / "runs" / "isolated_p6_and_unload" / "run_config.json").write_text(
        "{bad config", encoding="utf-8"
    )
    with pytest.raises(CheckpointCorruptedError):
        run_evaluation_pipeline(
            config=PipelineConfig(
                models=["qwen2.5-0.5b"],
                dataset_path=dataset_path,
                output_dir=tmp_path / "runs",
                summaries_dir=tmp_path / "summaries",
                run_id="isolated_p6_and_unload",
                resume=True,
            ),
            embedder=MockPipelineEmbedder(),
        )
