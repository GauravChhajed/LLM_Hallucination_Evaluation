"""
Provider-agnostic judge model client supporting hosted APIs and local Hugging Face models.

Handles missing API keys, timeouts, rate limits, API failures, and local loading errors
explicitly without silently substituting another judge model.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import gc
from typing import Any, Callable, Dict, Optional

import requests
import torch

from models.model_config import GenerationConfig, ModelBackend, ModelSpec, get_model_spec
from src.generation.model_loader import (
    LoadedLocalModel,
    MissingAPIKeyError,
    ModelLoadError,
    load_local_hf_model,
    validate_api_credentials,
)
from src.generation.prompt_builder import PromptPayload, format_prompt_for_tokenizer
from src.judge.judge_config import JudgeConfig
from src.judge.judge_prompt import JudgePromptPayload
from src.utils.config import AppConfig, get_config


class JudgeClientError(RuntimeError):
    """Base error raised by judge model backends."""


class JudgeMissingAPIKeyError(JudgeClientError):
    """Raised when a hosted judge model lacks its required API key."""


class JudgeTimeoutError(JudgeClientError):
    """Raised when a judge API request times out."""


class JudgeRateLimitError(JudgeClientError):
    """Raised when a judge API responds with HTTP 429 rate limiting."""


class JudgeAPIError(JudgeClientError):
    """Raised on non-timeout, non-rate-limit API failures."""


@dataclass(frozen=True)
class JudgeRawResponse:
    """Raw unparsed judge model output paired with latency and backend metadata."""

    raw_text: str
    judge_model_id: str
    backend: str
    latency_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseJudgeClient(ABC):
    """Abstract interface implemented by real and mocked judge model clients."""

    judge_model_id: str

    @abstractmethod
    def evaluate_prompt(self, prompt: JudgePromptPayload) -> JudgeRawResponse:
        """Send a standardized prompt to the judge model and return raw text output."""

    def unload(self) -> None:
        """Release any held judge model resources (no-op for stateless backends)."""
        return None


class ModelBackedJudgeClient(BaseJudgeClient):
    """Unified judge client executing against ModelSpec (local HF or hosted API)."""

    def __init__(
        self,
        spec: ModelSpec,
        judge_config: Optional[JudgeConfig] = None,
        app_config: Optional[AppConfig] = None,
        preferred_device: str = "auto",
        local_files_only: bool = False,
        loaded_model: Optional[LoadedLocalModel] = None,
        http_post: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.spec = spec
        self.judge_model_id = spec.model_id
        self.judge_config = judge_config or JudgeConfig(judge_model_key_or_id=spec.key)
        self.app_config = app_config or get_config()
        self.preferred_device = preferred_device
        self.local_files_only = local_files_only
        self._loaded: Optional[LoadedLocalModel] = loaded_model
        self._http_post = http_post or requests.post

    def _ensure_local_loaded(self) -> LoadedLocalModel:
        if self._loaded is None:
            try:
                self._loaded = load_local_hf_model(
                    spec=self.spec,
                    preferred_device=self.preferred_device,
                    app_config=self.app_config,
                    local_files_only=self.local_files_only,
                )
            except ModelLoadError as exc:
                raise JudgeClientError(str(exc)) from exc
        return self._loaded

    def _run_local(self, prompt: JudgePromptPayload) -> JudgeRawResponse:
        start = time.perf_counter()
        loaded = self._ensure_local_loaded()

        # Reuse prompt formatting helper cleanly via PromptPayload adapter
        adapter = PromptPayload(
            question=prompt.user_prompt,
            context="",
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            plain_text_prompt=prompt.plain_text_prompt,
            messages=prompt.messages,
            has_context=prompt.has_context,
            template_name=prompt.prompt_version,
        )
        formatted = format_prompt_for_tokenizer(
            payload=adapter,
            tokenizer=loaded.tokenizer,
            architecture=self.spec.architecture,
        )

        try:
            max_input_len = max(256, int(self.spec.context_window) - int(self.judge_config.max_new_tokens))
            encoded = loaded.tokenizer(
                formatted,
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

            input_len = int(encoded["input_ids"].shape[-1])
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": self.judge_config.max_new_tokens,
                "do_sample": self.judge_config.temperature > 0.0,
            }
            if gen_kwargs["do_sample"]:
                gen_kwargs["temperature"] = float(self.judge_config.temperature)

            pad_id = getattr(loaded.tokenizer, "pad_token_id", None) or getattr(
                loaded.tokenizer, "eos_token_id", None
            )
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id

            with torch.no_grad():
                output_ids = loaded.model.generate(**encoded, **gen_kwargs)

            seq = output_ids[0]
            new_tokens = seq if self.spec.architecture == "seq2seq_lm" else seq[input_len:]
            raw_text = (loaded.tokenizer.decode(new_tokens, skip_special_tokens=True) or "").strip()
        except Exception as exc:
            raise JudgeClientError(f"Local judge generation failed for '{self.spec.model_id}': {exc}") from exc

        elapsed = round(time.perf_counter() - start, 4)
        return JudgeRawResponse(
            raw_text=raw_text,
            judge_model_id=self.spec.model_id,
            backend=self.spec.backend.value,
            latency_seconds=elapsed,
            metadata={"device": loaded.device, "dtype": loaded.dtype},
        )

    def _run_api(self, prompt: JudgePromptPayload) -> JudgeRawResponse:
        start = time.perf_counter()
        try:
            api_key = validate_api_credentials(self.spec, app_config=self.app_config)
        except MissingAPIKeyError as exc:
            raise JudgeMissingAPIKeyError(str(exc)) from exc

        try:
            if self.spec.backend == ModelBackend.GEMINI_API:
                base = (
                    self.spec.api_base_url
                    or "https://generativelanguage.googleapis.com/v1beta"
                ).rstrip("/")
                url = f"{base}/models/{self.spec.model_id}:generateContent?key={api_key}"
                payload = {
                    "systemInstruction": {"parts": [{"text": prompt.system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt.user_prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": self.judge_config.max_new_tokens,
                        "temperature": float(self.judge_config.temperature),
                        "responseMimeType": "application/json",
                    },
                }
                resp = self._http_post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.judge_config.timeout_seconds,
                )
            else:
                if not self.spec.api_base_url:
                    raise JudgeAPIError(f"Judge model '{self.spec.key}' has no api_base_url configured.")
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.spec.model_id,
                    "messages": prompt.messages,
                    "max_tokens": self.judge_config.max_new_tokens,
                    "temperature": float(self.judge_config.temperature),
                }
                resp = self._http_post(
                    self.spec.api_base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.judge_config.timeout_seconds,
                )
        except requests.Timeout as exc:
            raise JudgeTimeoutError(
                f"Judge API request timed out after {self.judge_config.timeout_seconds}s "
                f"for '{self.spec.model_id}'."
            ) from exc
        except requests.RequestException as exc:
            raise JudgeAPIError(f"Judge API network failure for '{self.spec.model_id}': {exc}") from exc

        status_code = getattr(resp, "status_code", 200)
        if status_code == 429:
            raise JudgeRateLimitError(
                f"Judge API rate limit exceeded (HTTP 429) for '{self.spec.model_id}'."
            )
        try:
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            raise JudgeAPIError(
                f"Judge API error (HTTP {status_code}) for '{self.spec.model_id}': {exc}"
            ) from exc

        if self.spec.backend == ModelBackend.GEMINI_API:
            candidates = body.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            raw_text = "".join(str(p.get("text", "")) for p in parts).strip()
        else:
            choices = body.get("choices", [])
            raw_text = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""

        elapsed = round(time.perf_counter() - start, 4)
        return JudgeRawResponse(
            raw_text=raw_text,
            judge_model_id=self.spec.model_id,
            backend=self.spec.backend.value,
            latency_seconds=elapsed,
            metadata={"http_status": status_code},
        )

    def evaluate_prompt(self, prompt: JudgePromptPayload) -> JudgeRawResponse:
        """Execute prompt evaluation against the configured local or hosted judge backend."""
        if self.spec.is_local:
            return self._run_local(prompt)
        return self._run_api(prompt)

    def unload(self) -> None:
        """Unload any cached local judge model weights from memory."""
        if self._loaded is not None:
            self._loaded = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def create_judge_client(
    judge_config: Optional[JudgeConfig] = None,
    app_config: Optional[AppConfig] = None,
    preferred_device: str = "auto",
    local_files_only: bool = False,
) -> ModelBackedJudgeClient:
    """
    Create a ModelBackedJudgeClient without silently substituting unknown models.
    """
    cfg = app_config or get_config()
    j_cfg = judge_config or JudgeConfig(judge_model_key_or_id=cfg.evaluation.default_judge_model)
    spec = get_model_spec(j_cfg.judge_model_key_or_id)
    return ModelBackedJudgeClient(
        spec=spec,
        judge_config=j_cfg,
        app_config=cfg,
        preferred_device=preferred_device,
        local_files_only=local_files_only,
    )
