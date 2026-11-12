"""
Model configuration and registry for local Hugging Face and free-tier API models.

This module decouples model selection from the generation and evaluation pipeline
so models can be swapped or added without modifying downstream evaluation logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelBackend(str, Enum):
    """Supported model execution backends."""

    HUGGINGFACE_LOCAL = "huggingface_local"
    HUGGINGFACE_API = "huggingface_api"
    GROQ_API = "groq_api"
    GEMINI_API = "gemini_api"
    OPENROUTER_API = "openrouter_api"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass(frozen=True)
class GenerationConfig:
    """Deterministic-by-default generation hyperparameters shared across models."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    repetition_penalty: float = 1.05
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Serialize generation parameters to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ModelSpec:
    """Descriptor for a configurable LLM in the comparison framework."""

    key: str
    display_name: str
    model_id: str
    backend: ModelBackend
    architecture: str = "causal_lm"  # "causal_lm", "seq2seq_lm", or "chat_api"
    approx_vram_gb: float = 0.0
    recommended_dtype: str = "float16"
    api_key_env: Optional[str] = None
    api_base_url: Optional[str] = None
    context_window: int = 4096
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def is_local(self) -> bool:
        """Return True if the model executes locally on CPU/GPU."""
        return self.backend == ModelBackend.HUGGINGFACE_LOCAL

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model specification to a plain dictionary."""
        data = asdict(self)
        data["backend"] = self.backend.value
        data["is_local"] = self.is_local
        return data


# Pre-configured model catalog tailored for 4 GB VRAM (GTX 1650) + 16 GB RAM,
# alongside optional free-tier API models.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "qwen2.5-1.5b": ModelSpec(
        key="qwen2.5-1.5b",
        display_name="Qwen 2.5 1.5B Instruct (Local)",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        backend=ModelBackend.HUGGINGFACE_LOCAL,
        architecture="causal_lm",
        approx_vram_gb=3.1,
        recommended_dtype="float16",
        context_window=32768,
        description="Strong instruction-following 1.5B model that fits within 4 GB VRAM in FP16.",
        tags=["local", "recommended", "generator", "judge"],
    ),
    "qwen2.5-0.5b": ModelSpec(
        key="qwen2.5-0.5b",
        display_name="Qwen 2.5 0.5B Instruct (Local)",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=ModelBackend.HUGGINGFACE_LOCAL,
        architecture="causal_lm",
        approx_vram_gb=1.1,
        recommended_dtype="float16",
        context_window=32768,
        description="Ultra-lightweight 0.5B causal model for fast baseline comparison.",
        tags=["local", "fast", "generator"],
    ),
    "tinyllama-1.1b": ModelSpec(
        key="tinyllama-1.1b",
        display_name="TinyLlama 1.1B Chat v1.0 (Local)",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        backend=ModelBackend.HUGGINGFACE_LOCAL,
        architecture="causal_lm",
        approx_vram_gb=2.2,
        recommended_dtype="float16",
        context_window=2048,
        description="Compact 1.1B Llama-architecture chat model suitable for hallucination contrast.",
        tags=["local", "generator"],
    ),
    "smollm2-1.7b": ModelSpec(
        key="smollm2-1.7b",
        display_name="SmolLM2 1.7B Instruct (Local)",
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        backend=ModelBackend.HUGGINGFACE_LOCAL,
        architecture="causal_lm",
        approx_vram_gb=3.4,
        recommended_dtype="float16",
        context_window=8192,
        description="Modern compact 1.7B instruction-tuned model.",
        tags=["local", "generator"],
    ),
    "flan-t5-base": ModelSpec(
        key="flan-t5-base",
        display_name="FLAN-T5 Base 250M (Local)",
        model_id="google/flan-t5-base",
        backend=ModelBackend.HUGGINGFACE_LOCAL,
        architecture="seq2seq_lm",
        approx_vram_gb=1.0,
        recommended_dtype="float32",
        context_window=512,
        description="Lightweight encoder-decoder model for fast extractive QA baselines.",
        tags=["local", "seq2seq", "fast"],
    ),
    "flan-t5-large": ModelSpec(
        key="flan-t5-large",
        display_name="FLAN-T5 Large 780M (Local)",
        model_id="google/flan-t5-large",
        backend=ModelBackend.HUGGINGFACE_LOCAL,
        architecture="seq2seq_lm",
        approx_vram_gb=1.6,
        recommended_dtype="float16",
        context_window=512,
        description="Encoder-decoder instruction model well-suited for context-grounded QA.",
        tags=["local", "seq2seq", "recommended", "generator"],
    ),
    # Optional free-tier API models (require API keys in .env)
    "hf-llama-3.1-8b": ModelSpec(
        key="hf-llama-3.1-8b",
        display_name="Llama 3.1 8B Instruct (HF Inference API)",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        backend=ModelBackend.HUGGINGFACE_API,
        architecture="chat_api",
        approx_vram_gb=0.0,
        api_key_env="HF_TOKEN",
        api_base_url="https://router.huggingface.co/v1/chat/completions",
        context_window=8192,
        description="Hosted 8B model via Hugging Face free Inference API (requires HF_TOKEN).",
        tags=["api", "hf", "generator", "judge"],
    ),
    "groq-llama-3.1-8b": ModelSpec(
        key="groq-llama-3.1-8b",
        display_name="Llama 3.1 8B Instant (Groq API)",
        model_id="llama-3.1-8b-instant",
        backend=ModelBackend.GROQ_API,
        architecture="chat_api",
        approx_vram_gb=0.0,
        api_key_env="GROQ_API_KEY",
        api_base_url="https://api.groq.com/openai/v1/chat/completions",
        context_window=8192,
        description="Fast hosted 8B model via Groq free tier (requires GROQ_API_KEY).",
        tags=["api", "groq", "generator", "judge"],
    ),
    "groq-llama-3.3-70b": ModelSpec(
        key="groq-llama-3.3-70b",
        display_name="Llama 3.3 70B Versatile (Groq API)",
        model_id="llama-3.3-70b-versatile",
        backend=ModelBackend.GROQ_API,
        architecture="chat_api",
        approx_vram_gb=0.0,
        api_key_env="GROQ_API_KEY",
        api_base_url="https://api.groq.com/openai/v1/chat/completions",
        context_window=32768,
        description="Strong 70B model via Groq free tier for high-quality LLM-as-a-Judge.",
        tags=["api", "groq", "judge"],
    ),
    "gemini-2.5-flash": ModelSpec(
        key="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash (Google AI Studio Free Tier)",
        model_id="gemini-2.5-flash",
        backend=ModelBackend.GEMINI_API,
        architecture="chat_api",
        approx_vram_gb=0.0,
        api_key_env="GEMINI_API_KEY",
        api_base_url="https://generativelanguage.googleapis.com/v1beta",
        context_window=32768,
        description="Google Gemini free-tier model via REST API (requires GEMINI_API_KEY).",
        tags=["api", "gemini", "generator", "judge"],
    ),
}


def get_model_spec(model_key_or_id: str) -> ModelSpec:
    """
    Look up a ModelSpec by registry key or Hugging Face/API model_id.

    Raises KeyError with an informative message if no matching model exists.
    """
    if model_key_or_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key_or_id]

    for spec in MODEL_REGISTRY.values():
        if spec.model_id == model_key_or_id:
            return spec

    available = ", ".join(sorted(MODEL_REGISTRY.keys()))
    raise KeyError(
        f"Unknown model '{model_key_or_id}'. Available registered keys: {available}. "
        "Use register_custom_model() to add a custom model."
    )


def register_custom_model(
    key: str,
    model_id: str,
    backend: ModelBackend | str = ModelBackend.HUGGINGFACE_LOCAL,
    architecture: str = "causal_lm",
    display_name: Optional[str] = None,
    approx_vram_gb: float = 0.0,
    recommended_dtype: str = "float16",
    api_key_env: Optional[str] = None,
    api_base_url: Optional[str] = None,
    description: str = "User-registered custom model.",
) -> ModelSpec:
    """Register a custom local or API-backed model into MODEL_REGISTRY at runtime."""
    resolved_backend = backend if isinstance(backend, ModelBackend) else ModelBackend(backend)
    spec = ModelSpec(
        key=key,
        display_name=display_name or f"{model_id} ({resolved_backend.value})",
        model_id=model_id,
        backend=resolved_backend,
        architecture=architecture,
        approx_vram_gb=approx_vram_gb,
        recommended_dtype=recommended_dtype,
        api_key_env=api_key_env,
        api_base_url=api_base_url,
        description=description,
        tags=["custom"],
    )
    MODEL_REGISTRY[key] = spec
    return spec


def list_available_models(
    include_api: bool = True,
    tag_filter: Optional[str] = None,
) -> List[ModelSpec]:
    """Return registered models, optionally filtering out API models or by tag."""
    specs = list(MODEL_REGISTRY.values())
    if not include_api:
        specs = [s for s in specs if s.is_local]
    if tag_filter:
        specs = [s for s in specs if tag_filter in s.tags]
    return specs
