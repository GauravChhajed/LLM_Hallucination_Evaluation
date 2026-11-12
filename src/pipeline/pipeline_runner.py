"""
End-to-end evaluation pipeline runner integrating dataset loading, sequential
response generation, Phase 5–7 evaluation stages, checkpointing, and aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from models.model_config import ModelSpec
from src.evaluation.semantic_similarity import BaseEmbeddingProvider
from src.generation.generator import BaseLLMGenerator, GenerationResult, create_generator
from src.hallucination.claim_verifier import BaseNLIScorer
from src.judge.judge_client import BaseJudgeClient
from src.pipeline.aggregation import aggregate_pipeline_results
from src.pipeline.checkpoint_manager import CheckpointManager
from src.pipeline.evaluation_orchestrator import EvaluationOrchestrator
from src.pipeline.pipeline_config import (
    PipelineConfig,
    compute_config_fingerprint,
    compute_dataset_hash,
    resolve_model_specs,
)
from src.pipeline.result_manager import build_combined_results, save_run_outputs
from src.utils.config import AppConfig, get_config, resolve_device
from src.utils.data_loader import load_dataset


@dataclass
class PipelineRunResult:
    """Complete result object returned by `run_evaluation_pipeline`."""

    run_id: str
    run_dir: Path
    combined_results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    metadata: Dict[str, Any]
    output_files: Dict[str, Path]


class EvaluationPipelineRunner:
    """Centralized orchestrator connecting Phases 3–7 into reproducible runs."""

    def __init__(
        self,
        config: PipelineConfig,
        app_config: Optional[AppConfig] = None,
        generator_factory: Optional[Callable[[ModelSpec, PipelineConfig], BaseLLMGenerator]] = None,
        embedder: Optional[BaseEmbeddingProvider] = None,
        nli_scorer: Optional[BaseNLIScorer] = None,
        judge_client: Optional[BaseJudgeClient] = None,
    ) -> None:
        self.config = config
        self.app_config = app_config or get_config(ensure_dirs=True)
        self._generator_factory = generator_factory
        self._embedder = embedder
        self._nli_scorer = nli_scorer
        self._judge_client = judge_client

    def _make_generator(self, spec: ModelSpec) -> BaseLLMGenerator:
        if self._generator_factory is not None:
            return self._generator_factory(spec, self.config)
        return create_generator(
            model_key_or_id=spec.key,
            generation_config=self.config.generation_config,
            preferred_device=self.config.preferred_device,
            app_config=self.app_config,
            local_files_only=self.config.local_files_only,
        )

    def _load_and_slice_dataset(self) -> pd.DataFrame:
        df, _ = load_dataset(self.config.dataset_path, validate=True, raise_on_error=True)
        if self.config.question_ids:
            wanted = set(self.config.question_ids)
            df = df[df["question_id"].isin(wanted)].reset_index(drop=True)
            if df.empty:
                raise ValueError("No dataset rows matched PipelineConfig.question_ids.")
        if self.config.max_questions is not None:
            if self.config.max_questions <= 0:
                raise ValueError("PipelineConfig.max_questions must be positive.")
            df = df.head(self.config.max_questions).reset_index(drop=True)
        return df

    def _should_skip_generation(self, existing: Optional[GenerationResult]) -> bool:
        if existing is None:
            return False
        if existing.status == "success":
            return True
        if not self.config.retry_failed:
            return True
        return False

    def run(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> PipelineRunResult:
        """Execute the full generation + evaluation + aggregation pipeline."""
        started_at_utc = datetime.now(timezone.utc).isoformat()
        model_specs = resolve_model_specs(self.config.models)
        model_ids_in_order = [s.model_id for s in model_specs]
        dataset_df = self._load_and_slice_dataset()
        dataset_hash = compute_dataset_hash(dataset_df)
        fingerprint = compute_config_fingerprint(
            self.config, dataset_hash=dataset_hash, model_specs=model_specs
        )

        run_id = self.config.resolved_run_id()
        run_dir = self.config.output_dir / run_id
        checkpoint = CheckpointManager(run_dir)
        resumed = self.config.resume and checkpoint.run_config_path.exists()
        checkpoint.initialize_or_validate(
            serialized_config=self.config.to_dict(),
            fingerprint=fingerprint,
            resume=self.config.resume,
        )

        try:
            device_resolved = resolve_device(self.config.preferred_device)
        except Exception:
            device_resolved = self.config.preferred_device

        # 1. Load generation checkpoint (if resuming)
        generation_map: Dict[Tuple[str, str], GenerationResult] = (
            checkpoint.load_generation_checkpoint() if resumed else {}
        )

        # 2. Sequential model response generation (prevents simultaneous multi-model RAM load)
        total_gen_steps = len(model_specs) * len(dataset_df)
        completed_steps = 0
        generation_skipped_count = 0

        for spec in model_specs:
            pending_rows = [
                row
                for _, row in dataset_df.iterrows()
                if not self._should_skip_generation(
                    generation_map.get((spec.model_id, str(row["question_id"])))
                )
            ]

            if not pending_rows:
                generation_skipped_count += len(dataset_df)
                completed_steps += len(dataset_df)
                continue

            generator: Optional[BaseLLMGenerator] = None
            try:
                generator = self._make_generator(spec)
                for _, row in dataset_df.iterrows():
                    qid = str(row["question_id"])
                    key = (spec.model_id, qid)
                    completed_steps += 1

                    if self._should_skip_generation(generation_map.get(key)):
                        generation_skipped_count += 1
                        if progress_callback is not None:
                            progress_callback("generation", completed_steps, total_gen_steps)
                        continue

                    question = str(row["question"])
                    context = str(row.get("context", "") or "")
                    try:
                        res = generator.generate(
                            question_id=qid,
                            question=question,
                            context=context,
                            system_prompt=self.config.system_prompt,
                        )
                    except Exception as exc:
                        res = GenerationResult(
                            question_id=qid,
                            model_key=spec.key,
                            model_id=spec.model_id,
                            backend=spec.backend.value,
                            question=question,
                            context=context,
                            prompt_used="",
                            generated_answer="",
                            generation_time_seconds=0.0,
                            status="error",
                            error_message=f"Unexpected generation failure: {exc}",
                        )

                    generation_map[key] = res
                    if self.config.save_incremental:
                        checkpoint.save_generation_checkpoint(generation_map)
                    if progress_callback is not None:
                        progress_callback("generation", completed_steps, total_gen_steps)
            except Exception as exc:
                # If generator initialization fails, record explicit error per pending question
                for row in pending_rows:
                    qid = str(row["question_id"])
                    key = (spec.model_id, qid)
                    generation_map[key] = GenerationResult(
                        question_id=qid,
                        model_key=spec.key,
                        model_id=spec.model_id,
                        backend=spec.backend.value,
                        question=str(row["question"]),
                        context=str(row.get("context", "") or ""),
                        prompt_used="",
                        generated_answer="",
                        generation_time_seconds=0.0,
                        status="error",
                        error_message=f"Model initialization failed: {exc}",
                    )
                checkpoint.save_generation_checkpoint(generation_map)
            finally:
                if (
                    generator is not None
                    and self.config.unload_models_after_generation
                    and self.config.sequential_model_execution
                ):
                    generator.unload()
                generator = None

        checkpoint.save_generation_checkpoint(generation_map)

        # 3. Independent evaluation stages (Phases 5, 6, and optional 7)
        orchestrator = EvaluationOrchestrator(
            config=self.config,
            app_config=self.app_config,
            embedder=self._embedder,
            nli_scorer=self._nli_scorer,
            judge_client=self._judge_client,
        )
        eval_map, hallucination_map, judge_map = orchestrator.run_evaluations(
            dataset_df=dataset_df,
            model_ids_in_order=model_ids_in_order,
            generation_map=generation_map,
            checkpoint_manager=checkpoint,
            progress_callback=progress_callback,
        )

        # 4. Assemble combined results & export final artifacts
        combined_results = build_combined_results(
            dataset_df=dataset_df,
            model_ids_in_order=model_ids_in_order,
            generation_map=generation_map,
            evaluation_map=eval_map,
            hallucination_map=hallucination_map,
            judge_map=judge_map,
            config=self.config,
        )
        summary = aggregate_pipeline_results(combined_results)
        output_files = save_run_outputs(
            run_id=run_id,
            run_dir=run_dir,
            summaries_dir=self.config.summaries_dir,
            config=self.config,
            dataset_hash=dataset_hash,
            model_specs_info=[s.to_dict() for s in model_specs],
            combined_results=combined_results,
            started_at_utc=started_at_utc,
            resumed_from_checkpoint=resumed,
            device_resolved=device_resolved,
            generation_skipped_from_checkpoint=generation_skipped_count,
        )
        with output_files["run_metadata_json"].open("r", encoding="utf-8") as fh:
            import json

            metadata = json.load(fh)

        return PipelineRunResult(
            run_id=run_id,
            run_dir=run_dir,
            combined_results=combined_results,
            summary=summary,
            metadata=metadata,
            output_files=output_files,
        )


def run_evaluation_pipeline(
    config: PipelineConfig,
    app_config: Optional[AppConfig] = None,
    generator_factory: Optional[Callable[[ModelSpec, PipelineConfig], BaseLLMGenerator]] = None,
    embedder: Optional[BaseEmbeddingProvider] = None,
    nli_scorer: Optional[BaseNLIScorer] = None,
    judge_client: Optional[BaseJudgeClient] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> PipelineRunResult:
    """Convenience entry point to execute the integrated evaluation pipeline."""
    runner = EvaluationPipelineRunner(
        config=config,
        app_config=app_config,
        generator_factory=generator_factory,
        embedder=embedder,
        nli_scorer=nli_scorer,
        judge_client=judge_client,
    )
    return runner.run(progress_callback=progress_callback)
