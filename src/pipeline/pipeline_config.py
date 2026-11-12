"""
Configuration, validation, and reproducibility fingerprinting for the end-to-end
LLM hallucination evaluation pipeline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from models.model_config import GenerationConfig, ModelSpec, get_model_spec
from src.evaluation.correctness import CorrectnessConfig
from src.generation.prompt_builder import DEFAULT_SYSTEM_PROMPT, STANDARD_EVAL_TEMPLATE
from src.hallucination.claim_verifier import ClaimVerificationConfig
from src.judge.judge_config import JudgeConfig
from src.utils.config import PROJECT_ROOT


@dataclass
class PipelineConfig:
    """Complete configuration for an integrated multi-model evaluation run."""

    models: List[str]
    dataset_path: Path = PROJECT_ROOT / "data" / "evaluation_dataset.csv"
    output_dir: Path = PROJECT_ROOT / "results" / "runs"
    summaries_dir: Path = PROJECT_ROOT / "results" / "summaries"
    run_id: Optional[str] = None
    max_questions: Optional[int] = None
    question_ids: Optional[List[str]] = None
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    prompt_template_name: str = STANDARD_EVAL_TEMPLATE.name
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    preferred_device: str = "auto"
    sequential_model_execution: bool = True
    unload_models_after_generation: bool = True
    enable_reference_evaluation: bool = True
    enable_hallucination_detection: bool = True
    enable_judge: bool = False
    judge_config: Optional[JudgeConfig] = None
    correctness_config: CorrectnessConfig = field(default_factory=CorrectnessConfig)
    verification_config: ClaimVerificationConfig = field(default_factory=ClaimVerificationConfig)
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model_id: Optional[str] = None
    retrieval_top_k: int = 2
    retrieval_min_similarity: float = 0.30
    resume: bool = False
    retry_failed: bool = True
    save_incremental: bool = True
    local_files_only: bool = False

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
        self.summaries_dir = Path(self.summaries_dir)
        if self.enable_judge and self.judge_config is None:
            self.judge_config = JudgeConfig()

    def resolved_run_id(self) -> str:
        """Return explicit run_id or generate a deterministic-format timestamped run_id."""
        if self.run_id and self.run_id.strip():
            return self.run_id.strip()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short = hashlib.sha1(
            json.dumps(
                {
                    "models": self.models,
                    "dataset": str(self.dataset_path),
                    "max_q": self.max_questions,
                    "ts": ts,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:6]
        return f"run_{ts}_{short}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-safe dictionary."""
        return {
            "models": list(self.models),
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "summaries_dir": str(self.summaries_dir),
            "run_id": self.run_id,
            "max_questions": self.max_questions,
            "question_ids": list(self.question_ids) if self.question_ids else None,
            "generation_config": self.generation_config.to_dict(),
            "prompt_template_name": self.prompt_template_name,
            "system_prompt": self.system_prompt,
            "preferred_device": self.preferred_device,
            "sequential_model_execution": self.sequential_model_execution,
            "unload_models_after_generation": self.unload_models_after_generation,
            "enable_reference_evaluation": self.enable_reference_evaluation,
            "enable_hallucination_detection": self.enable_hallucination_detection,
            "enable_judge": self.enable_judge,
            "judge_config": self.judge_config.to_dict() if self.judge_config else None,
            "correctness_config": asdict(self.correctness_config),
            "verification_config": asdict(self.verification_config),
            "embedding_model_id": self.embedding_model_id,
            "nli_model_id": self.nli_model_id,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_min_similarity": self.retrieval_min_similarity,
            "resume": self.resume,
            "retry_failed": self.retry_failed,
            "save_incremental": self.save_incremental,
            "local_files_only": self.local_files_only,
        }


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 digest of dataset records for reproducibility."""
    canonical = df.sort_values("question_id").to_dict(orient="records") if "question_id" in df.columns else df.to_dict(orient="records")
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_model_specs(model_keys_or_ids: List[str]) -> List[ModelSpec]:
    """Validate and resolve all configured model keys/IDs without silent fallbacks."""
    if not model_keys_or_ids:
        raise ValueError("PipelineConfig.models must contain at least one model.")
    specs = [get_model_spec(m) for m in model_keys_or_ids]
    seen_ids: set[str] = set()
    for spec in specs:
        if spec.model_id in seen_ids:
            raise ValueError(
                f"Duplicate model_id '{spec.model_id}' in PipelineConfig.models. "
                "Each evaluated model in a run must have a distinct model_id."
            )
        seen_ids.add(spec.model_id)
    return specs


def compute_config_fingerprint(
    config: PipelineConfig,
    dataset_hash: str,
    model_specs: Optional[List[ModelSpec]] = None,
) -> Dict[str, Any]:
    """
    Build a compatibility fingerprint used to verify checkpoint resume safety.
    """
    specs = model_specs or resolve_model_specs(config.models)
    return {
        "dataset_hash": dataset_hash,
        "max_questions": config.max_questions,
        "question_ids": list(config.question_ids) if config.question_ids else None,
        "models": [
            {"key": s.key, "model_id": s.model_id, "backend": s.backend.value}
            for s in specs
        ],
        "prompt_template_name": config.prompt_template_name,
        "system_prompt": config.system_prompt,
        "generation_config": config.generation_config.to_dict(),
        "enable_reference_evaluation": config.enable_reference_evaluation,
        "enable_hallucination_detection": config.enable_hallucination_detection,
        "enable_judge": config.enable_judge,
        "judge_model": config.judge_config.judge_model_key_or_id if (config.enable_judge and config.judge_config) else None,
        "judge_prompt_version": config.judge_config.prompt_version if (config.enable_judge and config.judge_config) else None,
        "embedding_model_id": config.embedding_model_id,
        "nli_model_id": config.nli_model_id,
    }
