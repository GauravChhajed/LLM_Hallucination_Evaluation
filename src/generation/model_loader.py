"""
Model and tokenizer loading utilities for local Hugging Face and API backends.

Handles device resolution, CPU-safe dtype selection, credential validation,
and informative failure reporting without silently substituting models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from models.model_config import ModelBackend, ModelSpec
from src.utils.config import AppConfig, get_config, resolve_device


class ModelLoadError(RuntimeError):
    """Raised when a model, tokenizer, or required backend credential cannot be loaded."""


class MissingAPIKeyError(ModelLoadError):
    """Raised when an API-backed model is selected without its required API key."""


@dataclass
class LoadedLocalModel:
    """Container holding a loaded local Hugging Face model, tokenizer, and metadata."""

    spec: ModelSpec
    model: Any
    tokenizer: Any
    device: str
    dtype: str
    load_time_seconds: float


def resolve_torch_dtype(recommended_dtype: str, device: str) -> torch.dtype:
    """
    Select a safe PyTorch dtype for the target compute device.

    On CPU, defaults to float32 because many CPU kernels do not support float16.
    """
    if device == "cpu":
        return torch.float32

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get((recommended_dtype or "float32").strip().lower(), torch.float32)


def validate_api_credentials(spec: ModelSpec, app_config: Optional[AppConfig] = None) -> str:
    """
    Verify that the API key required by an API-backed ModelSpec is configured.

    Raises MissingAPIKeyError with setup instructions if missing.
    """
    if spec.is_local:
        raise ValueError(f"Model '{spec.key}' is local and does not require an API key.")

    cfg = app_config or get_config()
    if not spec.api_key_env:
        return ""

    api_key = cfg.get_api_key(spec.api_key_env)
    if not api_key:
        raise MissingAPIKeyError(
            f"Model '{spec.display_name}' ({spec.model_id}) requires environment variable "
            f"'{spec.api_key_env}', which is not set. Add it to your .env file before running."
        )
    return api_key


def load_local_hf_model(
    spec: ModelSpec,
    preferred_device: str = "auto",
    app_config: Optional[AppConfig] = None,
    local_files_only: bool = False,
    tokenizer_loader: Optional[Callable[..., Any]] = None,
    model_loader: Optional[Callable[..., Any]] = None,
) -> LoadedLocalModel:
    """
    Load a local Hugging Face tokenizer and model according to `spec`.

    Supports dependency injection (`tokenizer_loader`, `model_loader`) for testing
    without network calls or model weight downloads.
    """
    if spec.backend != ModelBackend.HUGGINGFACE_LOCAL:
        raise ModelLoadError(
            f"Cannot load '{spec.key}' locally because its backend is '{spec.backend.value}'."
        )

    cfg = app_config or get_config()
    cache_dir = str(cfg.paths.hf_cache_dir)

    try:
        device = resolve_device(preferred_device)
    except Exception as exc:
        raise ModelLoadError(f"Device resolution failed for '{spec.key}': {exc}") from exc

    torch_dtype = resolve_torch_dtype(spec.recommended_dtype, device)
    start_time = time.perf_counter()

    try:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        tok_fn = tokenizer_loader or AutoTokenizer.from_pretrained
        tokenizer = tok_fn(
            spec.model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if model_loader is not None:
            mdl_fn = model_loader
        elif spec.architecture == "seq2seq_lm":
            mdl_fn = AutoModelForSeq2SeqLM.from_pretrained
        else:
            mdl_fn = AutoModelForCausalLM.from_pretrained

        model = mdl_fn(
            spec.model_id,
            cache_dir=cache_dir,
            dtype=torch_dtype,
            local_files_only=local_files_only,
        )

        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load local Hugging Face model '{spec.model_id}' "
            f"(key='{spec.key}', device='{device}'): {exc}"
        ) from exc

    elapsed = round(time.perf_counter() - start_time, 4)
    return LoadedLocalModel(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=str(torch_dtype).replace("torch.", ""),
        load_time_seconds=elapsed,
    )
