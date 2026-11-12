"""Unit tests for the Phase 4 LLM response-generation module using mocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from models.model_config import GenerationConfig, ModelBackend, ModelSpec
from src.generation import (
    APIBasedGenerator,
    DEFAULT_SYSTEM_PROMPT,
    GenerationResult,
    HuggingFaceLocalGenerator,
    LoadedLocalModel,
    MissingAPIKeyError,
    ModelLoadError,
    PromptTemplate,
    STANDARD_EVAL_TEMPLATE,
    build_qa_prompt,
    create_generator,
    format_prompt_for_tokenizer,
    generate_paired_responses,
    load_generation_results,
    load_local_hf_model,
    resolve_torch_dtype,
    save_generation_results,
    validate_api_credentials,
)
from src.utils.config import get_config


class MockEncodedBatch(dict):
    """Dictionary mimicking Hugging Face BatchEncoding with `.to(device)`."""

    def to(self, device: str) -> "MockEncodedBatch":
        return MockEncodedBatch({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})


class MockTokenizer:
    """Lightweight mock tokenizer supporting chat templates and decoding."""

    def __init__(self, decoded_text: str = "Mock answer.", with_chat_template: bool = True) -> None:
        self.pad_token = None
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.chat_template = "mock-template" if with_chat_template else None
        self._decoded_text = decoded_text

    def apply_chat_template(
        self, messages: List[Dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        joined = " | ".join(f"{m['role']}: {m['content']}" for m in messages)
        return f"<|chat|>{joined}<|assistant|>"

    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = True, max_length: int = 512) -> MockEncodedBatch:
        return MockEncodedBatch(
            {
                "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        )

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        return self._decoded_text


class MockModel:
    """Lightweight mock model returning deterministic token tensors."""

    def __init__(self, output_tokens: List[int], raise_on_generate: bool = False) -> None:
        self._output_tokens = output_tokens
        self._raise_on_generate = raise_on_generate
        self.last_generate_kwargs: Dict[str, Any] = {}

    def to(self, device: str) -> "MockModel":
        return self

    def eval(self) -> "MockModel":
        return self

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.last_generate_kwargs = kwargs
        if self._raise_on_generate:
            raise RuntimeError("Simulated OOM during generation.")
        return torch.tensor([self._output_tokens], dtype=torch.long)


def _local_spec(architecture: str = "causal_lm") -> ModelSpec:
    return ModelSpec(
        key="mock-local",
        display_name="Mock Local Model",
        model_id="org/mock-local-model",
        backend=ModelBackend.HUGGINGFACE_LOCAL,
        architecture=architecture,
        recommended_dtype="float16",
        context_window=2048,
    )


def test_prompt_builder_with_and_without_context() -> None:
    """Verify neutral prompt construction, non-coaching default, and template extensibility."""
    # Ensure default instruction does not coach hallucination test behaviors
    lower_sys = DEFAULT_SYSTEM_PROMPT.lower()
    for forbidden_coaching in ("false premise", "ambiguous", "uncertainty", "refuse", "guessing"):
        assert forbidden_coaching not in lower_sys

    p_no_ctx = build_qa_prompt(question="What is the capital of France?", context="")
    assert p_no_ctx.has_context is False
    assert p_no_ctx.template_name == STANDARD_EVAL_TEMPLATE.name
    assert "Context:" not in p_no_ctx.user_prompt
    assert "What is the capital of France?" in p_no_ctx.user_prompt
    assert len(p_no_ctx.messages) == 2

    p_ctx = build_qa_prompt(
        question="How many satellites launched?",
        context="Phase 2 added 18 satellites.",
    )
    assert p_ctx.has_context is True
    assert "Context:\nPhase 2 added 18 satellites." in p_ctx.user_prompt
    assert "Question:\nHow many satellites launched?" in p_ctx.user_prompt
    assert p_ctx.user_prompt.endswith("Answer:")

    custom_tpl = PromptTemplate(
        name="custom_ablation",
        system_prompt="Custom system prompt.",
        context_format="[CTX] {context} [Q] {question}",
        no_context_format="[Q] {question}",
    )
    p_custom = build_qa_prompt(question="Test?", template=custom_tpl)
    assert p_custom.template_name == "custom_ablation"
    assert p_custom.user_prompt == "[Q] Test?"

    tok = MockTokenizer(with_chat_template=True)
    assert format_prompt_for_tokenizer(p_ctx, tok, architecture="causal_lm").startswith("<|chat|>")
    assert format_prompt_for_tokenizer(p_ctx, tok, architecture="seq2seq_lm") == p_ctx.plain_text_prompt

    with pytest.raises(ValueError):
        build_qa_prompt(question="   ")


def test_model_loader_cpu_dtype_and_error_handling() -> None:
    """Verify CPU dtype safety and explicit errors on missing models or API keys."""
    assert resolve_torch_dtype("float16", "cpu") == torch.float32
    assert resolve_torch_dtype("float16", "cuda") == torch.float16

    spec = _local_spec()

    def failing_tokenizer(*args: Any, **kwargs: Any) -> Any:
        raise OSError("Model weights not found locally.")

    with pytest.raises(ModelLoadError) as exc_info:
        load_local_hf_model(spec, preferred_device="cpu", tokenizer_loader=failing_tokenizer)
    assert "org/mock-local-model" in str(exc_info.value)

    api_spec = ModelSpec(
        key="mock-groq",
        display_name="Mock Groq",
        model_id="llama-3.1-8b-instant",
        backend=ModelBackend.GROQ_API,
        architecture="chat_api",
        api_key_env="NON_EXISTENT_TEST_API_KEY_XYZ",
    )
    with pytest.raises(MissingAPIKeyError) as key_exc:
        validate_api_credentials(api_spec)
    assert "NON_EXISTENT_TEST_API_KEY_XYZ" in str(key_exc.value)


def test_local_generator_causal_and_seq2seq_structure_and_timing() -> None:
    """Verify structured GenerationResult, token slicing, deterministic kwargs, and timing."""
    causal_spec = _local_spec(architecture="causal_lm")
    # Input has 3 tokens ([10, 11, 12]); model returns 6 tokens -> 3 newly generated tokens
    mock_tok = MockTokenizer(decoded_text="Canberra is the capital.")
    mock_mdl = MockModel(output_tokens=[10, 11, 12, 101, 102, 103])
    loaded = LoadedLocalModel(
        spec=causal_spec,
        model=mock_mdl,
        tokenizer=mock_tok,
        device="cpu",
        dtype="float32",
        load_time_seconds=0.01,
    )
    gen_cfg = GenerationConfig(max_new_tokens=64, temperature=0.0, do_sample=False, seed=42)
    generator = HuggingFaceLocalGenerator(spec=causal_spec, generation_config=gen_cfg, loaded_model=loaded)

    res = generator.generate(question_id="Q001", question="Capital of Australia?", context="")
    assert isinstance(res, GenerationResult)
    assert res.is_success is True
    assert res.status == "success"
    assert res.question_id == "Q001"
    assert res.model_id == "org/mock-local-model"
    assert res.generated_answer == "Canberra is the capital."
    assert res.generation_time_seconds >= 0.0
    assert res.metadata["device"] == "cpu"
    assert res.metadata["dtype"] == "float32"
    assert res.metadata["input_tokens"] == 3
    assert res.metadata["output_tokens"] == 3
    assert mock_mdl.last_generate_kwargs["do_sample"] is False
    assert "temperature" not in mock_mdl.last_generate_kwargs

    # Seq2Seq models decode full sequence (not sliced by input length)
    seq2seq_spec = _local_spec(architecture="seq2seq_lm")
    loaded_seq2seq = LoadedLocalModel(
        spec=seq2seq_spec,
        model=MockModel(output_tokens=[201, 202]),
        tokenizer=MockTokenizer(decoded_text="408"),
        device="cpu",
        dtype="float32",
        load_time_seconds=0.01,
    )
    seq_gen = HuggingFaceLocalGenerator(spec=seq2seq_spec, generation_config=gen_cfg, loaded_model=loaded_seq2seq)
    seq_res = seq_gen.generate(question_id="Q031", question="What is 17 * 24?")
    assert seq_res.generated_answer == "408"
    assert seq_res.metadata["output_tokens"] == 2


def test_empty_response_and_generation_error_handling() -> None:
    """Verify empty model outputs and runtime exceptions return clean error statuses."""
    spec = _local_spec()

    empty_loaded = LoadedLocalModel(
        spec=spec,
        model=MockModel(output_tokens=[10, 11, 12]),
        tokenizer=MockTokenizer(decoded_text="   "),
        device="cpu",
        dtype="float32",
        load_time_seconds=0.01,
    )
    empty_gen = HuggingFaceLocalGenerator(spec=spec, loaded_model=empty_loaded)
    empty_res = empty_gen.generate(question_id="Q010", question="Who painted The Starry Night?")
    assert empty_res.status == "empty_response"
    assert empty_res.is_success is False
    assert empty_res.generated_answer == ""
    assert empty_res.error_message is not None

    err_loaded = LoadedLocalModel(
        spec=spec,
        model=MockModel(output_tokens=[], raise_on_generate=True),
        tokenizer=MockTokenizer(decoded_text=""),
        device="cpu",
        dtype="float32",
        load_time_seconds=0.01,
    )
    err_gen = HuggingFaceLocalGenerator(spec=spec, loaded_model=err_loaded)
    err_res = err_gen.generate(question_id="Q011", question="Smallest country?")
    assert err_res.status == "error"
    assert err_res.is_success is False
    assert "Simulated OOM" in (err_res.error_message or "")


def test_api_generator_and_factory_configuration(monkeypatch) -> None:
    """Verify API generator request handling, missing key error, and factory lookup."""
    cfg = get_config()
    local_gen = create_generator("qwen2.5-0.5b", app_config=cfg)
    assert isinstance(local_gen, HuggingFaceLocalGenerator)

    api_gen = create_generator("groq-llama-3.1-8b", app_config=cfg)
    assert isinstance(api_gen, APIBasedGenerator)

    with pytest.raises(KeyError):
        create_generator("nonexistent-model-key", app_config=cfg)

    # Test mocked API success
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-key")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "Mock API answer."}}]}
    mock_resp.raise_for_status.return_value = None
    mock_post = MagicMock(return_value=mock_resp)

    mocked_api = APIBasedGenerator(
        spec=api_gen.spec,
        generation_config=GenerationConfig(max_new_tokens=100),
        app_config=cfg,
        http_post=mock_post,
    )
    res = mocked_api.generate(question_id="Q001", question="Capital of Australia?")
    assert res.is_success is True
    assert res.generated_answer == "Mock API answer."
    assert mock_post.called


def test_paired_dataset_generation_and_persistence(tmp_path) -> None:
    """Verify paired Model A / Model B generation receives identical inputs and saves cleanly."""
    df = pd.DataFrame(
        [
            {"question_id": "Q001", "question": "Capital of Australia?", "context": ""},
            {
                "question_id": "Q066",
                "question": "How many satellites in Phase 2?",
                "context": "Phase 2 added 18 satellites.",
            },
        ]
    )

    spec_a = _local_spec()
    spec_b = ModelSpec(
        key="mock-b",
        display_name="Mock Model B",
        model_id="org/mock-b",
        backend=ModelBackend.HUGGINGFACE_LOCAL,
        architecture="seq2seq_lm",
    )

    gen_a = HuggingFaceLocalGenerator(
        spec=spec_a,
        loaded_model=LoadedLocalModel(
            spec=spec_a,
            model=MockModel([10, 11, 12, 50]),
            tokenizer=MockTokenizer("Model A answer."),
            device="cpu",
            dtype="float32",
            load_time_seconds=0.01,
        ),
    )
    gen_b = HuggingFaceLocalGenerator(
        spec=spec_b,
        loaded_model=LoadedLocalModel(
            spec=spec_b,
            model=MockModel([60, 61]),
            tokenizer=MockTokenizer("Model B answer."),
            device="cpu",
            dtype="float32",
            load_time_seconds=0.01,
        ),
    )

    res_a, res_b = generate_paired_responses(df, gen_a, gen_b)
    assert [r.question_id for r in res_a] == ["Q001", "Q066"]
    assert [r.question_id for r in res_b] == ["Q001", "Q066"]
    assert [r.question for r in res_a] == [r.question for r in res_b]
    assert [r.context for r in res_a] == [r.context for r in res_b]

    paths = save_generation_results(res_a, output_dir=tmp_path, run_name="mock_run_a")
    assert paths["json"].exists() and paths["csv"].exists()

    reloaded = load_generation_results(paths["json"])
    assert len(reloaded) == 2
    assert reloaded[0].generated_answer == "Model A answer."
