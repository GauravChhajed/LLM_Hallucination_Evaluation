"""Unit tests for Phase 7 LLM-as-a-Judge evaluation framework using mocks."""

from __future__ import annotations

import json
from typing import Any, Callable, List
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from models.model_config import ModelBackend, ModelSpec
from src.evaluation.evaluator import EvaluationRecord
from src.generation import GenerationResult
from src.hallucination import (
    FaithfulnessMetrics,
    HallucinationDetectionRecord,
)
from src.judge import (
    JUDGE_PROMPT_VERSION,
    NOT_APPLICABLE_LABEL,
    BaseJudgeClient,
    JudgeConfig,
    JudgeEvaluationRecord,
    JudgeEvaluator,
    JudgePromptPayload,
    JudgeRawResponse,
    ModelBackedJudgeClient,
    aggregate_judge_metrics,
    build_judge_prompt,
    compute_overall_score,
    compute_repeated_judging_consistency,
    load_judge_results,
    parse_and_validate_judge_output,
    save_judge_results,
)
from src.utils.config import get_config


class MockJudgeClient(BaseJudgeClient):
    """Configurable mock judge client capturing prompts and returning scripted outputs."""

    def __init__(
        self,
        responder: Callable[[JudgePromptPayload], str] | None = None,
        judge_model_id: str = "mock-judge-llm",
    ) -> None:
        self.judge_model_id = judge_model_id
        self._responder = responder
        self.captured_prompts: List[JudgePromptPayload] = []

    def evaluate_prompt(self, prompt: JudgePromptPayload) -> JudgeRawResponse:
        self.captured_prompts.append(prompt)
        text = self._responder(prompt) if self._responder else "{}"
        return JudgeRawResponse(
            raw_text=text,
            judge_model_id=self.judge_model_id,
            backend="mock_judge",
            latency_seconds=0.05,
        )


def _valid_json_str(
    rel: int = 5,
    fact: int = 5,
    comp: int = 5,
    ctx: int | str = 5,
    clarity: int = 5,
    conf: float = 0.92,
) -> str:
    return json.dumps(
        {
            "relevance_score": rel,
            "factual_correctness_score": fact,
            "completeness_score": comp,
            "contextual_faithfulness_score": ctx,
            "clarity_score": clarity,
            "dimension_explanations": {
                "relevance": "Directly answers the question.",
                "factual_correctness": "Matches the reference answer.",
                "completeness": "Covers all essential information.",
                "contextual_faithfulness": "Grounded in context or not applicable.",
                "clarity": "Clear and concise.",
            },
            "judge_confidence": conf,
            "overall_summary": "Accurate and well-structured response.",
        }
    )


def test_prompt_construction_and_blind_evaluation() -> None:
    """Verify prompt captures version/context/counterfactuals and hides model identity."""
    prompt = build_judge_prompt(
        question_id="Q079",
        question="What is the boiling point of water?",
        reference_answer="72 degrees Celsius.",
        generated_answer="Water boils at 72 degrees Celsius.",
        context="In this simulation, pure water boils at 72 degrees Celsius.",
        is_counterfactual_context=True,
    )
    assert prompt.prompt_version == JUDGE_PROMPT_VERSION
    assert prompt.has_context is True
    assert prompt.is_counterfactual_context is True
    assert "counterfactual" in prompt.user_prompt.lower()

    client = MockJudgeClient(responder=lambda p: _valid_json_str(ctx=5))
    evaluator = JudgeEvaluator(judge_client=client)
    rec = evaluator.evaluate_single(
        question_id="Q079",
        evaluated_model_id="secret-qwen-model-xyz",
        question="What is the boiling point of water?",
        reference_answer="72 degrees Celsius.",
        generated_answer="Water boils at 72 degrees Celsius.",
        context="In this simulation, pure water boils at 72 degrees Celsius.",
        hallucination_test_type="counterfactual_context",
    )
    assert rec.evaluation_status == "evaluated"
    assert rec.metadata["is_counterfactual_context"] is True
    # Model identity must NEVER appear in the judge prompt
    sent_prompt = client.captured_prompts[0].plain_text_prompt
    assert "secret-qwen-model-xyz" not in sent_prompt


def test_json_parsing_fenced_and_validation_failures() -> None:
    """Verify fenced JSON parsing plus invalid JSON, missing fields, and out-of-range rejection."""
    fenced = f"```json\n{_valid_json_str(rel=4, fact=5, comp=4, ctx=NOT_APPLICABLE_LABEL, clarity=5, conf=0.88)}\n```"
    parsed_ok = parse_and_validate_judge_output(fenced, has_context=False)
    assert parsed_ok.is_valid is True
    assert parsed_ok.contextual_faithfulness_score == NOT_APPLICABLE_LABEL
    assert parsed_ok.judge_confidence == pytest.approx(0.88)

    # Invalid JSON
    bad_json = parse_and_validate_judge_output("Not JSON at all", has_context=False)
    assert bad_json.is_valid is False

    # Missing required field
    missing_field = json.dumps({"relevance_score": 5, "factual_correctness_score": 5})
    parsed_missing = parse_and_validate_judge_output(missing_field, has_context=False)
    assert parsed_missing.is_valid is False
    assert any("Missing required JSON fields" in err for err in parsed_missing.errors)

    # Out-of-range score (6) and out-of-range confidence (1.5)
    out_of_range = _valid_json_str(rel=6, conf=1.5)
    parsed_oor = parse_and_validate_judge_output(out_of_range, has_context=True)
    assert parsed_oor.is_valid is False
    assert any("out of range [1, 5]" in err for err in parsed_oor.errors)
    assert any("judge_confidence" in err for err in parsed_oor.errors)

    # NOT_APPLICABLE rejected when context WAS provided
    bad_na = _valid_json_str(ctx=NOT_APPLICABLE_LABEL)
    parsed_bad_na = parse_and_validate_judge_output(bad_na, has_context=True)
    assert parsed_bad_na.is_valid is False


def test_api_errors_missing_key_timeout_and_rate_limit(monkeypatch) -> None:
    """Verify API error types map to explicit statuses without silent model substitution."""
    cfg = get_config()
    api_spec = ModelSpec(
        key="mock-groq-judge",
        display_name="Mock Groq Judge",
        model_id="llama-3.3-70b-versatile",
        backend=ModelBackend.GROQ_API,
        architecture="chat_api",
        api_key_env="TEST_JUDGE_GROQ_KEY_MISSING",
        api_base_url="https://api.groq.com/openai/v1/chat/completions",
    )

    # 1. Missing API key
    client_no_key = ModelBackedJudgeClient(spec=api_spec, app_config=cfg)
    ev_no_key = JudgeEvaluator(judge_client=client_no_key)
    r1 = ev_no_key.evaluate_single("Q001", "model-a", "Q?", "Ref", "Ans")
    assert r1.evaluation_status == "missing_api_key"
    assert r1.overall_score is None

    # 2. Timeout
    monkeypatch.setenv("TEST_JUDGE_GROQ_KEY_MISSING", "fake-key")

    def timeout_post(*args: Any, **kwargs: Any) -> Any:
        raise requests.Timeout("Read timed out.")

    ev_timeout = JudgeEvaluator(
        judge_client=ModelBackedJudgeClient(spec=api_spec, app_config=cfg, http_post=timeout_post)
    )
    r2 = ev_timeout.evaluate_single("Q001", "model-a", "Q?", "Ref", "Ans")
    assert r2.evaluation_status == "judge_timeout"

    # 3. Rate limit (HTTP 429)
    rate_resp = MagicMock()
    rate_resp.status_code = 429
    ev_rate = JudgeEvaluator(
        judge_client=ModelBackedJudgeClient(
            spec=api_spec, app_config=cfg, http_post=lambda *a, **kw: rate_resp
        )
    )
    r3 = ev_rate.evaluate_single("Q001", "model-a", "Q?", "Ref", "Ans")
    assert r3.evaluation_status == "judge_rate_limited"

    # 4. General HTTP 500 error
    err_resp = MagicMock()
    err_resp.status_code = 500
    err_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    ev_err = JudgeEvaluator(
        judge_client=ModelBackedJudgeClient(
            spec=api_spec, app_config=cfg, http_post=lambda *a, **kw: err_resp
        )
    )
    r4 = ev_err.evaluate_single("Q001", "model-a", "Q?", "Ref", "Ans")
    assert r4.evaluation_status == "judge_api_error"


def test_no_context_contextual_refusal_and_invalid_output_preservation() -> None:
    """Verify no-context NOT_APPLICABLE weighting, appropriate refusals, and raw audit logs."""

    def scripted(prompt: JudgePromptPayload) -> str:
        if prompt.question_id == "Q001":
            return _valid_json_str(rel=5, fact=5, comp=5, ctx=NOT_APPLICABLE_LABEL, clarity=5, conf=0.95)
        if prompt.question_id == "Q074":
            # Appropriate refusal on unanswerable context question
            return _valid_json_str(rel=5, fact=5, comp=5, ctx=5, clarity=5, conf=0.90)
        return "BROKEN NON-JSON OUTPUT"

    evaluator = JudgeEvaluator(judge_client=MockJudgeClient(responder=scripted))

    # No-context question -> NOT_APPLICABLE contextual faithfulness, overall re-normalized to 5.0
    no_ctx = evaluator.evaluate_single(
        question_id="Q001",
        evaluated_model_id="model-a",
        question="What is the capital of Australia?",
        reference_answer="Canberra.",
        generated_answer="Canberra is the capital of Australia.",
        context="",
    )
    assert no_ctx.contextual_faithfulness_score == NOT_APPLICABLE_LABEL
    assert no_ctx.overall_score == pytest.approx(5.0)
    assert no_ctx.judge_confidence == pytest.approx(0.95)

    # Appropriate refusal on unanswerable context -> scored fairly at 5
    refusal = evaluator.evaluate_single(
        question_id="Q074",
        evaluated_model_id="model-a",
        question="What was the construction cost of the Kestrel Bridge?",
        reference_answer="The context does not provide the construction cost.",
        generated_answer="The passage does not state the construction cost.",
        context="The Kestrel Bridge spans 1.4 km and opened in 2018.",
        hallucination_test_type="unanswerable_from_context",
        expected_behavior="acknowledge_insufficient_info",
    )
    assert refusal.evaluation_status == "evaluated"
    assert refusal.contextual_faithfulness_score == 5
    assert refusal.factual_correctness_score == 5

    # Invalid judge output preserves raw_judge_response and sets invalid_judge_output
    broken = evaluator.evaluate_single(
        question_id="Q099",
        evaluated_model_id="model-a",
        question="Q?",
        reference_answer="Ref",
        generated_answer="Ans",
    )
    assert broken.evaluation_status == "invalid_judge_output"
    assert broken.raw_judge_response == "BROKEN NON-JSON OUTPUT"
    assert broken.overall_score is None


def test_multi_model_pipeline_integration_consistency_and_persistence(tmp_path) -> None:
    """
    Verify multi-model evaluation consuming Phases 4, 5, and 6 outputs independently,
    score aggregation, repeated-judging consistency, and JSON/CSV export.
    """
    dataset_df = pd.DataFrame(
        [
            {
                "question_id": "Q066",
                "question": "How many satellites in Phase 2?",
                "context": "Phase 2 added 18 radar satellites.",
                "reference_answer": "18 radar satellites.",
                "category": "Context-based Question Answering",
                "hallucination_test_type": "context_supported_precision",
                "expected_behavior": "context_grounded_answer",
            }
        ]
    )

    gen_a = [
        GenerationResult(
            question_id="Q066",
            model_key="a",
            model_id="org/model-a",
            backend="huggingface_local",
            question="How many satellites in Phase 2?",
            context="Phase 2 added 18 radar satellites.",
            prompt_used="...",
            generated_answer="Phase 2 added 18 radar satellites.",
            generation_time_seconds=0.3,
            status="success",
        )
    ]
    gen_b = [
        GenerationResult(
            question_id="Q066",
            model_key="b",
            model_id="org/model-b",
            backend="huggingface_local",
            question="How many satellites in Phase 2?",
            context="Phase 2 added 18 radar satellites.",
            prompt_used="...",
            generated_answer="Phase 2 added 24 radar satellites.",
            generation_time_seconds=0.25,
            status="success",
        )
    ]

    p5_a = [
        EvaluationRecord(
            question_id="Q066",
            model_id="org/model-a",
            generated_answer="Phase 2 added 18 radar satellites.",
            reference_answer="18 radar satellites.",
            exact_match=0.0,
            normalized_exact_match=0.0,
            token_precision=0.6,
            token_recall=1.0,
            token_f1=0.75,
            semantic_similarity=0.92,
            evaluation_status="evaluated",
        )
    ]
    p6_a = [
        HallucinationDetectionRecord(
            question_id="Q066",
            model_id="org/model-a",
            category="Context-based Question Answering",
            hallucination_test_type="context_supported_precision",
            expected_behavior="context_grounded_answer",
            has_context=True,
            is_counterfactual_context=False,
            generated_answer="Phase 2 added 18 radar satellites.",
            context="Phase 2 added 18 radar satellites.",
            reference_answer="18 radar satellites.",
            claims=[],
            faithfulness=FaithfulnessMetrics(
                question_id="Q066",
                model_id="org/model-a",
                has_context=True,
                supported_count=1,
                contradicted_count=0,
                unsupported_count=0,
                total_factual_claims=1,
                faithfulness_ratio=1.0,
                contradiction_rate=0.0,
                unsupported_claim_rate=0.0,
                context_hallucination_rate=0.0,
                status="computed",
            ),
            hallucination_detected=False,
            hallucination_types=[],
            context_hallucination_rate=0.0,
            detection_status="computed",
            explanation="Supported.",
        )
    ]

    def blind_scorer(prompt: JudgePromptPayload) -> str:
        if "24 radar" in prompt.user_prompt:
            return _valid_json_str(rel=5, fact=1, comp=3, ctx=1, clarity=5, conf=0.91)
        return _valid_json_str(rel=5, fact=5, comp=5, ctx=5, clarity=5, conf=0.94)

    evaluator = JudgeEvaluator(judge_client=MockJudgeClient(responder=blind_scorer))
    multi = evaluator.evaluate_multiple_models(
        {"model-a": gen_a, "model-b": gen_b},
        dataset_df=dataset_df,
        phase5_map={"model-a": p5_a},
        phase6_map={"model-a": p6_a},
    )

    rec_a = multi["model-a"][0]
    rec_b = multi["model-b"][0]
    assert rec_a.overall_score == pytest.approx(5.0)
    assert rec_b.factual_correctness_score == 1
    assert rec_b.contextual_faithfulness_score == 1
    assert rec_a.metadata["phase5_semantic_similarity"] == pytest.approx(0.92)
    assert rec_a.metadata["phase6_faithfulness_ratio"] == pytest.approx(1.0)

    agg = aggregate_judge_metrics([rec_a, rec_b])
    assert agg["org/model-a"]["avg_overall_score"] == pytest.approx(5.0)
    assert agg["org/model-b"]["avg_factual_correctness_score"] == pytest.approx(1.0)

    consistency = compute_repeated_judging_consistency([rec_a, rec_a])
    assert consistency["overall_score_std"] == pytest.approx(0.0)

    paths = save_judge_results([rec_a, rec_b], output_dir=tmp_path, run_name="judge_eval")
    assert paths["json"].exists() and paths["csv"].exists()
    reloaded = load_judge_results(paths["json"])
    assert len(reloaded) == 2
    assert isinstance(reloaded[0], JudgeEvaluationRecord)
    df_csv = pd.read_csv(paths["csv"])
    assert len(df_csv) == 2
    assert "judge_confidence" in df_csv.columns
