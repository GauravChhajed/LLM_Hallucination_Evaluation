"""LLM-as-a-Judge evaluation package."""

from src.judge.judge_client import (
    BaseJudgeClient,
    JudgeAPIError,
    JudgeClientError,
    JudgeMissingAPIKeyError,
    JudgeRateLimitError,
    JudgeRawResponse,
    JudgeTimeoutError,
    ModelBackedJudgeClient,
    create_judge_client,
)
from src.judge.judge_config import (
    DEFAULT_DIMENSION_WEIGHTS,
    DIMENSIONS,
    JUDGE_PROMPT_VERSION,
    RUBRIC_DEFINITIONS,
    JudgeConfig,
)
from src.judge.judge_evaluator import (
    JudgeEvaluationRecord,
    JudgeEvaluator,
    judge_records_to_dataframe,
    load_judge_results,
    save_judge_results,
)
from src.judge.judge_metrics import (
    aggregate_judge_metrics,
    compute_overall_score,
    compute_repeated_judging_consistency,
)
from src.judge.judge_parser import (
    NOT_APPLICABLE_LABEL,
    JudgeParseResult,
    extract_json_object,
    parse_and_validate_judge_output,
)
from src.judge.judge_prompt import JUDGE_SYSTEM_PROMPT, JudgePromptPayload, build_judge_prompt

__all__ = [
    "BaseJudgeClient",
    "DEFAULT_DIMENSION_WEIGHTS",
    "DIMENSIONS",
    "JUDGE_PROMPT_VERSION",
    "JUDGE_SYSTEM_PROMPT",
    "JudgeAPIError",
    "JudgeClientError",
    "JudgeConfig",
    "JudgeEvaluationRecord",
    "JudgeEvaluator",
    "JudgeMissingAPIKeyError",
    "JudgeParseResult",
    "JudgePromptPayload",
    "JudgeRateLimitError",
    "JudgeRawResponse",
    "JudgeTimeoutError",
    "ModelBackedJudgeClient",
    "NOT_APPLICABLE_LABEL",
    "RUBRIC_DEFINITIONS",
    "aggregate_judge_metrics",
    "build_judge_prompt",
    "compute_overall_score",
    "compute_repeated_judging_consistency",
    "create_judge_client",
    "extract_json_object",
    "judge_records_to_dataframe",
    "load_judge_results",
    "parse_and_validate_judge_output",
    "save_judge_results",
]
