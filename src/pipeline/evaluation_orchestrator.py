"""
Evaluation stage orchestrator coordinating Phase 5 (reference evaluation),
Phase 6 (faithfulness & hallucination detection), and Phase 7 (optional LLM-as-a-Judge)
with independent stage isolation and incremental checkpointing.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import pandas as pd

from src.evaluation.correctness import ReferenceCorrectnessEvaluator
from src.evaluation.evaluator import EvaluationRecord, ResponseEvaluator
from src.evaluation.semantic_similarity import BaseEmbeddingProvider, SentenceTransformerEmbedder
from src.generation.generator import GenerationResult
from src.hallucination.claim_verifier import BaseNLIScorer, CrossEncoderNLIScorer
from src.hallucination.faithfulness import compute_faithfulness_metrics
from src.hallucination.hallucination_detector import (
    HallucinationDetectionRecord,
    HallucinationDetector,
)
from src.judge.judge_client import BaseJudgeClient, create_judge_client
from src.judge.judge_config import JudgeConfig
from src.judge.judge_evaluator import JudgeEvaluationRecord, JudgeEvaluator
from src.pipeline.checkpoint_manager import CheckpointManager
from src.pipeline.pipeline_config import PipelineConfig
from src.utils.config import AppConfig, get_config


class EvaluationOrchestrator:
    """Runs Phase 5, 6, and 7 evaluators independently with checkpoint support."""

    def __init__(
        self,
        config: PipelineConfig,
        app_config: Optional[AppConfig] = None,
        embedder: Optional[BaseEmbeddingProvider] = None,
        nli_scorer: Optional[BaseNLIScorer] = None,
        judge_client: Optional[BaseJudgeClient] = None,
    ) -> None:
        self.config = config
        self.app_config = app_config or get_config()
        self._embedder = embedder
        self._nli_scorer = nli_scorer
        self._judge_client = judge_client

    def _get_embedder(self) -> BaseEmbeddingProvider:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(
                model_name=self.config.embedding_model_id,
                preferred_device=self.config.preferred_device,
                app_config=self.app_config,
                local_files_only=self.config.local_files_only,
            )
        return self._embedder

    def _get_nli_scorer(self) -> Optional[BaseNLIScorer]:
        if self._nli_scorer is not None:
            return self._nli_scorer
        if self.config.nli_model_id:
            self._nli_scorer = CrossEncoderNLIScorer(
                model_name=self.config.nli_model_id,
                preferred_device=self.config.preferred_device,
                app_config=self.app_config,
                local_files_only=self.config.local_files_only,
            )
        return self._nli_scorer

    def _should_skip_p5(self, rec: Optional[EvaluationRecord]) -> bool:
        if rec is None:
            return False
        if rec.evaluation_status in {"evaluated", "evaluated_without_semantic"}:
            return True
        if not self.config.retry_failed:
            return True
        return False

    def _should_skip_p6(self, rec: Optional[HallucinationDetectionRecord]) -> bool:
        if rec is None:
            return False
        if rec.detection_status not in {"detection_error", "generation_error"}:
            return True
        if not self.config.retry_failed:
            return True
        return False

    def _should_skip_p7(self, rec: Optional[JudgeEvaluationRecord]) -> bool:
        if rec is None:
            return False
        if rec.evaluation_status == "evaluated":
            return True
        if not self.config.retry_failed:
            return True
        return False

    def run_evaluations(
        self,
        dataset_df: pd.DataFrame,
        model_ids_in_order: list[str],
        generation_map: Mapping[Tuple[str, str], GenerationResult],
        checkpoint_manager: CheckpointManager,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> tuple[
        Dict[Tuple[str, str], EvaluationRecord],
        Dict[Tuple[str, str], HallucinationDetectionRecord],
        Dict[Tuple[str, str], JudgeEvaluationRecord],
    ]:
        """Execute all enabled evaluation stages independently and incrementally."""
        eval_map: Dict[Tuple[str, str], EvaluationRecord] = (
            checkpoint_manager.load_evaluation_checkpoint()
            if (self.config.resume and self.config.enable_reference_evaluation)
            else {}
        )
        hallucination_map: Dict[Tuple[str, str], HallucinationDetectionRecord] = (
            checkpoint_manager.load_hallucination_checkpoint()
            if (self.config.resume and self.config.enable_hallucination_detection)
            else {}
        )
        judge_map: Dict[Tuple[str, str], JudgeEvaluationRecord] = (
            checkpoint_manager.load_judge_checkpoint()
            if (self.config.resume and self.config.enable_judge)
            else {}
        )

        # Build ordered work pairs aligned with dataset & model ordering
        pairs: list[tuple[pd.Series, str]] = []
        for _, row in dataset_df.iterrows():
            for model_id in model_ids_in_order:
                pairs.append((row, model_id))
        total_pairs = len(pairs)

        # Stage 1: Phase 5 Reference-based Evaluation
        if self.config.enable_reference_evaluation:
            ref_evaluator = ResponseEvaluator(
                embedder=self._get_embedder(),
                correctness_evaluator=ReferenceCorrectnessEvaluator(
                    config=self.config.correctness_config
                ),
            )
            for idx, (row, model_id) in enumerate(pairs, start=1):
                qid = str(row["question_id"])
                key = (model_id, qid)
                if self._should_skip_p5(eval_map.get(key)):
                    continue
                gen = generation_map.get(key)
                if gen is None:
                    continue

                try:
                    rec = ref_evaluator.evaluate_single(
                        question_id=qid,
                        model_id=model_id,
                        generated_answer=gen.generated_answer,
                        reference_answer=str(row.get("reference_answer", "") or ""),
                        generation_status=gen.status,
                        question_type=str(row.get("category", "") or ""),
                        expected_behavior=str(row.get("expected_behavior", "") or ""),
                        hallucination_test_type=str(row.get("hallucination_test_type", "") or ""),
                        extra_metadata={
                            "model_key": gen.model_key,
                            "difficulty": str(row.get("difficulty", "") or ""),
                            "generation_time_seconds": gen.generation_time_seconds,
                        },
                    )
                except Exception as exc:
                    rec = EvaluationRecord(
                        question_id=qid,
                        model_id=model_id,
                        generated_answer=gen.generated_answer,
                        reference_answer=str(row.get("reference_answer", "") or ""),
                        exact_match=0.0,
                        normalized_exact_match=0.0,
                        token_precision=0.0,
                        token_recall=0.0,
                        token_f1=0.0,
                        semantic_similarity=None,
                        evaluation_status="evaluation_error",
                        evaluation_metadata={"error_message": str(exc)},
                    )
                eval_map[key] = rec
                if self.config.save_incremental:
                    checkpoint_manager.save_evaluation_checkpoint(eval_map)
                if progress_callback is not None:
                    progress_callback("reference_evaluation", idx, total_pairs)

            checkpoint_manager.save_evaluation_checkpoint(eval_map)

        # Stage 2: Phase 6 Faithfulness & Hallucination Detection
        if self.config.enable_hallucination_detection:
            detector = HallucinationDetector(
                embedder=self._get_embedder(),
                nli_scorer=self._get_nli_scorer(),
                verification_config=self.config.verification_config,
                retrieval_top_k=self.config.retrieval_top_k,
                retrieval_min_similarity=self.config.retrieval_min_similarity,
            )
            for idx, (row, model_id) in enumerate(pairs, start=1):
                qid = str(row["question_id"])
                key = (model_id, qid)
                if self._should_skip_p6(hallucination_map.get(key)):
                    continue
                gen = generation_map.get(key)
                if gen is None:
                    continue

                ctx = str(row.get("context", "") or "")
                ref = str(row.get("reference_answer", "") or "")
                cat = str(row.get("category", "") or "")
                h_type = str(row.get("hallucination_test_type", "") or "")
                exp_beh = str(row.get("expected_behavior", "") or "")

                try:
                    h_rec = detector.detect_single(
                        question_id=qid,
                        model_id=model_id,
                        generated_answer=gen.generated_answer,
                        context=ctx,
                        reference_answer=ref,
                        category=cat,
                        hallucination_test_type=h_type,
                        expected_behavior=exp_beh,
                        generation_status=gen.status,
                    )
                except Exception as exc:
                    has_ctx = len(ctx.strip()) > 0
                    faith = compute_faithfulness_metrics(
                        qid, model_id, has_ctx, [], extraction_status="empty_response"
                    )
                    h_rec = HallucinationDetectionRecord(
                        question_id=qid,
                        model_id=model_id,
                        category=cat,
                        hallucination_test_type=h_type,
                        expected_behavior=exp_beh,
                        has_context=has_ctx,
                        is_counterfactual_context=h_type == "counterfactual_context",
                        generated_answer=gen.generated_answer,
                        context=ctx,
                        reference_answer=ref,
                        claims=[],
                        faithfulness=faith,
                        hallucination_detected=False,
                        hallucination_types=[],
                        context_hallucination_rate=None,
                        detection_status="detection_error",
                        explanation=f"Hallucination detection failed: {exc}",
                    )
                hallucination_map[key] = h_rec
                if self.config.save_incremental:
                    checkpoint_manager.save_hallucination_checkpoint(hallucination_map)
                if progress_callback is not None:
                    progress_callback("hallucination_detection", idx, total_pairs)

            checkpoint_manager.save_hallucination_checkpoint(hallucination_map)

        # Stage 3: Phase 7 Optional LLM-as-a-Judge
        if self.config.enable_judge:
            j_cfg = self.config.judge_config or JudgeConfig()
            active_client = self._judge_client or create_judge_client(
                judge_config=j_cfg,
                app_config=self.app_config,
                preferred_device=self.config.preferred_device,
                local_files_only=self.config.local_files_only,
            )
            judge_eval = JudgeEvaluator(judge_client=active_client, judge_config=j_cfg)

            try:
                for idx, (row, model_id) in enumerate(pairs, start=1):
                    qid = str(row["question_id"])
                    key = (model_id, qid)
                    if self._should_skip_p7(judge_map.get(key)):
                        continue
                    gen = generation_map.get(key)
                    if gen is None:
                        continue

                    extra: Dict[str, Any] = {
                        "generation_time_seconds": gen.generation_time_seconds,
                    }
                    if key in eval_map:
                        extra["phase5_semantic_similarity"] = eval_map[key].semantic_similarity
                        extra["phase5_token_f1"] = eval_map[key].token_f1
                    if key in hallucination_map:
                        extra["phase6_faithfulness_ratio"] = hallucination_map[key].faithfulness.faithfulness_ratio
                        extra["phase6_hallucination_detected"] = hallucination_map[key].hallucination_detected

                    j_rec = judge_eval.evaluate_single(
                        question_id=qid,
                        evaluated_model_id=model_id,
                        question=str(row.get("question", "") or ""),
                        reference_answer=str(row.get("reference_answer", "") or ""),
                        generated_answer=gen.generated_answer,
                        context=str(row.get("context", "") or ""),
                        category=str(row.get("category", "") or ""),
                        hallucination_test_type=str(row.get("hallucination_test_type", "") or ""),
                        expected_behavior=str(row.get("expected_behavior", "") or ""),
                        generation_status=gen.status,
                        extra_metadata=extra,
                    )
                    judge_map[key] = j_rec
                    if self.config.save_incremental:
                        checkpoint_manager.save_judge_checkpoint(judge_map)
                    if progress_callback is not None:
                        progress_callback("llm_judge", idx, total_pairs)
            finally:
                if self.config.unload_models_after_generation and hasattr(active_client, "unload"):
                    active_client.unload()

            checkpoint_manager.save_judge_checkpoint(judge_map)

        return eval_map, hallucination_map, judge_map
