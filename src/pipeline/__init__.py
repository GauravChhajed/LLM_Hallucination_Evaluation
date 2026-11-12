"""End-to-end LLM hallucination evaluation pipeline package."""

from src.pipeline.aggregation import aggregate_pipeline_results
from src.pipeline.checkpoint_manager import (
    CheckpointCorruptedError,
    CheckpointError,
    CheckpointManager,
    CheckpointMismatchError,
)
from src.pipeline.evaluation_orchestrator import EvaluationOrchestrator
from src.pipeline.pipeline_config import (
    PipelineConfig,
    compute_config_fingerprint,
    compute_dataset_hash,
    resolve_model_specs,
)
from src.pipeline.pipeline_runner import (
    EvaluationPipelineRunner,
    PipelineRunResult,
    run_evaluation_pipeline,
)
from src.pipeline.result_manager import (
    build_combined_results,
    combined_results_to_dataframe,
    save_run_outputs,
)

__all__ = [
    "CheckpointCorruptedError",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointMismatchError",
    "EvaluationOrchestrator",
    "EvaluationPipelineRunner",
    "PipelineConfig",
    "PipelineRunResult",
    "aggregate_pipeline_results",
    "build_combined_results",
    "combined_results_to_dataframe",
    "compute_config_fingerprint",
    "compute_dataset_hash",
    "resolve_model_specs",
    "run_evaluation_pipeline",
    "save_run_outputs",
]
