"""LLM response generation package."""

from src.generation.generator import (
    APIBasedGenerator,
    BaseLLMGenerator,
    GenerationResult,
    HuggingFaceLocalGenerator,
    create_generator,
    set_reproducible_seed,
)
from src.generation.model_loader import (
    LoadedLocalModel,
    MissingAPIKeyError,
    ModelLoadError,
    load_local_hf_model,
    resolve_torch_dtype,
    validate_api_credentials,
)
from src.generation.prompt_builder import (
    DEFAULT_SYSTEM_PROMPT,
    STANDARD_EVAL_TEMPLATE,
    PromptPayload,
    PromptTemplate,
    build_qa_prompt,
    format_prompt_for_tokenizer,
)
from src.generation.response_generator import (
    generate_paired_responses,
    generate_responses_for_dataset,
    load_generation_results,
    results_to_dataframe,
    save_generation_results,
)

__all__ = [
    "APIBasedGenerator",
    "BaseLLMGenerator",
    "DEFAULT_SYSTEM_PROMPT",
    "GenerationResult",
    "HuggingFaceLocalGenerator",
    "LoadedLocalModel",
    "MissingAPIKeyError",
    "ModelLoadError",
    "PromptPayload",
    "PromptTemplate",
    "STANDARD_EVAL_TEMPLATE",
    "build_qa_prompt",
    "create_generator",
    "format_prompt_for_tokenizer",
    "generate_paired_responses",
    "generate_responses_for_dataset",
    "load_generation_results",
    "load_local_hf_model",
    "resolve_torch_dtype",
    "results_to_dataframe",
    "save_generation_results",
    "set_reproducible_seed",
    "validate_api_credentials",
]
