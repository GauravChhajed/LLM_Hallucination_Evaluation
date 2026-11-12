"""
Centralized application configuration and environment management.

Loads `.env` values via python-dotenv, configures Hugging Face cache locations
(defaulting to D:\\hf_cache), resolves compute device settings, and provides
typed configuration objects to all downstream modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from models.model_config import GenerationConfig

# Project root is two levels above src/utils/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE: str = r"D:\hf_cache"


def _configure_environment(env_path: Optional[Path] = None) -> None:
    """
    Load environment variables from `.env` and ensure Hugging Face cache dirs
    are configured before `transformers` or `sentence_transformers` are imported.
    """
    target_env = env_path or (PROJECT_ROOT / ".env")
    if target_env.exists():
        load_dotenv(dotenv_path=target_env, override=False)

    hf_home = os.environ.get("HF_HOME") or DEFAULT_HF_CACHE
    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HF_HUB_CACHE", str(Path(hf_home) / "hub"))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME", str(Path(hf_home) / "sentence_transformers")
    )


_configure_environment()


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths used across data, cache, and results."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    default_dataset_path: Path = PROJECT_ROOT / "data" / "evaluation_dataset.csv"
    results_dir: Path = PROJECT_ROOT / "results"
    model_responses_dir: Path = PROJECT_ROOT / "results" / "model_responses"
    evaluation_reports_dir: Path = PROJECT_ROOT / "results" / "evaluation_reports"
    hf_cache_dir: Path = Path(os.environ.get("HF_HOME", DEFAULT_HF_CACHE))

    def ensure_directories(self) -> None:
        """Create required project output and cache directories if missing."""
        for path in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.results_dir,
            self.model_responses_dir,
            self.evaluation_reports_dir,
            self.hf_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configurable evaluation models, thresholds, and fairness settings."""

    embedding_model_id: str = field(
        default_factory=lambda: os.environ.get(
            "DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    nli_model_id: str = field(
        default_factory=lambda: os.environ.get(
            "DEFAULT_NLI_MODEL", "cross-encoder/nli-deberta-v3-small"
        )
    )
    default_judge_model: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_JUDGE_MODEL", "qwen2.5-1.5b")
    )
    support_similarity_threshold: float = 0.55
    contradiction_nli_threshold: float = 0.60
    entailment_nli_threshold: float = 0.55
    # Documented explicit weights if an optional weighted summary is displayed
    correctness_semantic_weight: float = 0.45
    correctness_key_info_weight: float = 0.35
    correctness_factual_weight: float = 0.20


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration container."""

    paths: PathConfig = field(default_factory=PathConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    default_model_a: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_MODEL_A", "qwen2.5-1.5b")
    )
    default_model_b: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_MODEL_B", "flan-t5-large")
    )
    preferred_device: str = field(
        default_factory=lambda: os.environ.get("DEVICE", "auto")
    )

    @staticmethod
    def get_api_key(env_var_name: Optional[str]) -> Optional[str]:
        """Safely retrieve an optional API key from the environment."""
        if not env_var_name:
            return None
        value = os.environ.get(env_var_name, "").strip()
        return value or None


def resolve_device(preferred_device: str = "auto") -> str:
    """
    Resolve the runtime compute device ('cuda' or 'cpu').

    Raises ValueError if 'cuda' is explicitly requested when CUDA is unavailable.
    """
    import torch

    pref = preferred_device.strip().lower()
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE='cuda' was requested, but torch.cuda.is_available() is False."
            )
        return "cuda"
    if pref == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device preference '{preferred_device}'. Use 'auto', 'cuda', or 'cpu'.")


def get_config(env_path: Optional[Path] = None, ensure_dirs: bool = False) -> AppConfig:
    """Load environment configuration and return an initialized AppConfig instance."""
    _configure_environment(env_path=env_path)
    cfg = AppConfig()
    if ensure_dirs:
        cfg.paths.ensure_directories()
    return cfg
