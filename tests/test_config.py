"""Basic Phase 2 smoke tests verifying environment, dependencies, PyTorch, and configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from models.model_config import (
    MODEL_REGISTRY,
    ModelBackend,
    get_model_spec,
    list_available_models,
    register_custom_model,
)
from src.utils.config import get_config, resolve_device


def test_virtual_environment_active() -> None:
    """Ensure tests are executing inside the project's .venv virtual environment."""
    assert sys.prefix != sys.base_prefix
    assert ".venv" in Path(sys.executable).parts


def test_core_dependencies_importable() -> None:
    """Verify all required ML, data, dashboard, and utility libraries import cleanly."""
    import accelerate
    import dotenv
    import huggingface_hub
    import numpy
    import pandas
    import plotly
    import requests
    import sentence_transformers
    import sklearn
    import streamlit
    import transformers

    assert all(
        mod is not None
        for mod in (
            accelerate,
            dotenv,
            huggingface_hub,
            numpy,
            pandas,
            plotly,
            requests,
            sentence_transformers,
            sklearn,
            streamlit,
            transformers,
        )
    )


def test_pytorch_tensor_ops_and_device_fallback() -> None:
    """Verify PyTorch tensor arithmetic works and device resolution handles CPU/CUDA."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    dot = torch.dot(a, b)
    assert float(dot.item()) == pytest.approx(32.0)

    resolved = resolve_device("auto")
    if torch.cuda.is_available():
        assert resolved == "cuda"
    else:
        assert resolved == "cpu"
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_config_and_hf_cache_loaded() -> None:
    """Verify .env configuration and D:\\hf_cache path setup."""
    cfg = get_config(ensure_dirs=True)
    assert str(cfg.paths.hf_cache_dir).lower() == r"d:\hf_cache"
    assert os.environ.get("HF_HOME", "").lower() == r"d:\hf_cache"
    assert cfg.paths.model_responses_dir.exists()
    assert cfg.paths.evaluation_reports_dir.exists()


def test_model_registry_switching_and_custom_registration() -> None:
    """Verify local and API models can be selected and custom models registered."""
    local_models = list_available_models(include_api=False)
    all_models = list_available_models(include_api=True)
    assert len(local_models) >= 4
    assert len(all_models) > len(local_models)

    qwen = get_model_spec("qwen2.5-1.5b")
    assert qwen.backend == ModelBackend.HUGGINGFACE_LOCAL
    assert qwen.is_local is True

    groq = get_model_spec("groq-llama-3.1-8b")
    assert groq.backend == ModelBackend.GROQ_API
    assert groq.is_local is False
    assert groq.api_key_env == "GROQ_API_KEY"

    custom_key = "test-custom-model"
    try:
        spec = register_custom_model(
            key=custom_key,
            model_id="org/custom-model",
            backend=ModelBackend.HUGGINGFACE_LOCAL,
        )
        assert get_model_spec(custom_key) == spec
    finally:
        MODEL_REGISTRY.pop(custom_key, None)
