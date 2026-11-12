"""
Unified LLM generation interface supporting local Hugging Face and API models.

Provides deterministic seeding, standardized structured outputs, timing metrics,
and graceful handling of loading, inference, and empty-response edge cases.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gc
from typing import Any, Dict, Optional

import numpy as np
import requests
import torch

from models.model_config import GenerationConfig, ModelBackend, ModelSpec, get_model_spec
from src.generation.model_loader import (
    LoadedLocalModel,
    ModelLoadError,
    load_local_hf_model,
    validate_api_credentials,
)
from src.generation.prompt_builder import build_qa_prompt, format_prompt_for_tokenizer
from src.utils.config import AppConfig, get_config


@dataclass
class GenerationResult:
    """Structured output returned for every single question-model inference."""

    question_id: str
    model_key: str
    model_id: str
    backend: str
    question: str
    context: str
    prompt_used: str
    generated_answer: str
    generation_time_seconds: float
    status: str  # "success", "empty_response", or "error"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Return True if generation succeeded with a non-empty answer."""
        return self.status == "success" and len(self.generated_answer.strip()) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the generation result into a flat-friendly dictionary."""
        return asdict(self)


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for deterministic generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed

        set_seed(seed)
    except Exception:
        pass


class BaseLLMGenerator(ABC):
    """Abstract base class for interchangeable LLM inference backends."""

    def __init__(
        self,
        spec: ModelSpec,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.spec = spec
        self.generation_config = generation_config or GenerationConfig()

    @abstractmethod
    def generate(
        self,
        question_id: str,
        question: str,
        context: str = "",
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate an answer for a single question and optional context."""

    def unload(self) -> None:
        """Release any held model/tokenizer memory (no-op for stateless backends)."""
        return None


class HuggingFaceLocalGenerator(BaseLLMGenerator):
    """Response generator backed by a local Hugging Face causal or seq2seq model."""

    def __init__(
        self,
        spec: ModelSpec,
        generation_config: Optional[GenerationConfig] = None,
        preferred_device: str = "auto",
        app_config: Optional[AppConfig] = None,
        loaded_model: Optional[LoadedLocalModel] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(spec=spec, generation_config=generation_config)
        self.preferred_device = preferred_device
        self.app_config = app_config or get_config()
        self.local_files_only = local_files_only
        self._loaded: Optional[LoadedLocalModel] = loaded_model

    def ensure_loaded(self) -> LoadedLocalModel:
        """Lazily load and cache the local model/tokenizer on first use."""
        if self._loaded is None:
            self._loaded = load_local_hf_model(
                spec=self.spec,
                preferred_device=self.preferred_device,
                app_config=self.app_config,
                local_files_only=self.local_files_only,
            )
        return self._loaded

    def unload(self) -> None:
        """Unload cached model and tokenizer weights from memory."""
        if self._loaded is not None:
            self._loaded = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build_generate_kwargs(self, tokenizer: Any) -> Dict[str, Any]:
        cfg = self.generation_config
        kwargs: Dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "repetition_penalty": cfg.repetition_penalty,
        }
        if cfg.do_sample:
            kwargs["temperature"] = max(float(cfg.temperature), 1e-5)
            kwargs["top_p"] = float(cfg.top_p)

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is not None:
            kwargs["pad_token_id"] = pad_id
        return kwargs

    def generate(
        self,
        question_id: str,
        question: str,
        context: str = "",
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        prompt_payload = build_qa_prompt(
            question=question,
            context=context,
            system_prompt=system_prompt,
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()

        try:
            loaded = self.ensure_loaded()
        except ModelLoadError as exc:
            elapsed = round(time.perf_counter() - start_time, 4)
            return GenerationResult(
                question_id=question_id,
                model_key=self.spec.key,
                model_id=self.spec.model_id,
                backend=self.spec.backend.value,
                question=prompt_payload.question,
                context=prompt_payload.context,
                prompt_used=prompt_payload.plain_text_prompt,
                generated_answer="",
                generation_time_seconds=elapsed,
                status="error",
                error_message=str(exc),
                metadata={"timestamp_utc": timestamp},
            )

        formatted_prompt = format_prompt_for_tokenizer(
            payload=prompt_payload,
            tokenizer=loaded.tokenizer,
            architecture=self.spec.architecture,
        )

        try:
            set_reproducible_seed(self.generation_config.seed)
            max_input_len = max(128, int(self.spec.context_window) - int(self.generation_config.max_new_tokens))
            encoded = loaded.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len,
            )
            if hasattr(encoded, "to"):
                encoded = encoded.to(loaded.device)
            elif isinstance(encoded, dict):
                encoded = {
                    k: v.to(loaded.device) if hasattr(v, "to") else v
                    for k, v in encoded.items()
                }

            input_ids = encoded["input_ids"]
            input_token_count = int(input_ids.shape[-1])
            gen_kwargs = self._build_generate_kwargs(loaded.tokenizer)

            with torch.no_grad():
                output_ids = loaded.model.generate(**encoded, **gen_kwargs)

            first_seq = output_ids[0]
            if self.spec.architecture == "seq2seq_lm":
                new_tokens = first_seq
            else:
                new_tokens = first_seq[input_token_count:]

            output_token_count = int(len(new_tokens))
            decoded = loaded.tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer_text = (decoded or "").strip()
            elapsed = round(time.perf_counter() - start_time, 4)

            status = "success" if answer_text else "empty_response"
            error_msg = (
                "Model returned an empty response." if status == "empty_response" else None
            )

            return GenerationResult(
                question_id=question_id,
                model_key=self.spec.key,
                model_id=self.spec.model_id,
                backend=self.spec.backend.value,
                question=prompt_payload.question,
                context=prompt_payload.context,
                prompt_used=formatted_prompt,
                generated_answer=answer_text,
                generation_time_seconds=elapsed,
                status=status,
                error_message=error_msg,
                metadata={
                    "device": loaded.device,
                    "dtype": loaded.dtype,
                    "architecture": self.spec.architecture,
                    "input_tokens": input_token_count,
                    "output_tokens": output_token_count,
                    "generation_config": self.generation_config.to_dict(),
                    "timestamp_utc": timestamp,
                },
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - start_time, 4)
            return GenerationResult(
                question_id=question_id,
                model_key=self.spec.key,
                model_id=self.spec.model_id,
                backend=self.spec.backend.value,
                question=prompt_payload.question,
                context=prompt_payload.context,
                prompt_used=formatted_prompt,
                generated_answer="",
                generation_time_seconds=elapsed,
                status="error",
                error_message=f"Generation failed: {exc}",
                metadata={
                    "device": loaded.device,
                    "dtype": loaded.dtype,
                    "generation_config": self.generation_config.to_dict(),
                    "timestamp_utc": timestamp,
                },
            )


class APIBasedGenerator(BaseLLMGenerator):
    """Response generator for free-tier HTTP chat completion APIs."""

    def __init__(
        self,
        spec: ModelSpec,
        generation_config: Optional[GenerationConfig] = None,
        app_config: Optional[AppConfig] = None,
        timeout_seconds: int = 60,
        http_post: Optional[Any] = None,
    ) -> None:
        super().__init__(spec=spec, generation_config=generation_config)
        self.app_config = app_config or get_config()
        self.timeout_seconds = timeout_seconds
        self._http_post = http_post or requests.post

    def _call_openai_compatible(self, api_key: str, prompt_payload: Any) -> str:
        if not self.spec.api_base_url:
            raise ValueError(f"Model '{self.spec.key}' has no api_base_url configured.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": prompt_payload.messages,
            "max_tokens": self.generation_config.max_new_tokens,
            "temperature": self.generation_config.temperature if self.generation_config.do_sample else 0.0,
            "top_p": self.generation_config.top_p,
        }
        response = self._http_post(
            self.spec.api_base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return (message.get("content") or "").strip()

    def _call_gemini_api(self, api_key: str, prompt_payload: Any) -> str:
        base = (self.spec.api_base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{base}/models/{self.spec.model_id}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": prompt_payload.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": prompt_payload.user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.generation_config.max_new_tokens,
                "temperature": self.generation_config.temperature if self.generation_config.do_sample else 0.0,
                "topP": self.generation_config.top_p,
            },
        }
        response = self._http_post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        candidates = body.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(str(part.get("text", "")) for part in parts).strip()

    def generate(
        self,
        question_id: str,
        question: str,
        context: str = "",
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        prompt_payload = build_qa_prompt(
            question=question,
            context=context,
            system_prompt=system_prompt,
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()

        try:
            api_key = validate_api_credentials(self.spec, app_config=self.app_config)
            if self.spec.backend == ModelBackend.GEMINI_API:
                answer_text = self._call_gemini_api(api_key, prompt_payload)
            else:
                answer_text = self._call_openai_compatible(api_key, prompt_payload)

            elapsed = round(time.perf_counter() - start_time, 4)
            status = "success" if answer_text else "empty_response"
            error_msg = "API returned an empty response." if status == "empty_response" else None

            return GenerationResult(
                question_id=question_id,
                model_key=self.spec.key,
                model_id=self.spec.model_id,
                backend=self.spec.backend.value,
                question=prompt_payload.question,
                context=prompt_payload.context,
                prompt_used=prompt_payload.plain_text_prompt,
                generated_answer=answer_text,
                generation_time_seconds=elapsed,
                status=status,
                error_message=error_msg,
                metadata={
                    "generation_config": self.generation_config.to_dict(),
                    "timestamp_utc": timestamp,
                },
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - start_time, 4)
            return GenerationResult(
                question_id=question_id,
                model_key=self.spec.key,
                model_id=self.spec.model_id,
                backend=self.spec.backend.value,
                question=prompt_payload.question,
                context=prompt_payload.context,
                prompt_used=prompt_payload.plain_text_prompt,
                generated_answer="",
                generation_time_seconds=elapsed,
                status="error",
                error_message=f"API generation failed: {exc}",
                metadata={
                    "generation_config": self.generation_config.to_dict(),
                    "timestamp_utc": timestamp,
                },
            )


def create_generator(
    model_key_or_id: str,
    generation_config: Optional[GenerationConfig] = None,
    preferred_device: str = "auto",
    app_config: Optional[AppConfig] = None,
    local_files_only: bool = False,
) -> BaseLLMGenerator:
    """
    Factory returning the appropriate generator backend for a configured model.

    Never silently falls back to a different model if the requested model is unknown.
    """
    spec = get_model_spec(model_key_or_id)
    cfg = app_config or get_config()

    if spec.backend == ModelBackend.HUGGINGFACE_LOCAL:
        return HuggingFaceLocalGenerator(
            spec=spec,
            generation_config=generation_config,
            preferred_device=preferred_device,
            app_config=cfg,
            local_files_only=local_files_only,
        )

    return APIBasedGenerator(
        spec=spec,
        generation_config=generation_config,
        app_config=cfg,
    )
