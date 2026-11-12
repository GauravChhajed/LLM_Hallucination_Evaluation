"""
Configuration, rubric definitions, and scoring weights for LLM-as-a-Judge evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

JUDGE_PROMPT_VERSION: str = "judge_rubric_v1.0"

DIMENSIONS: tuple[str, ...] = (
    "relevance",
    "factual_correctness",
    "completeness",
    "contextual_faithfulness",
    "clarity",
)

DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "relevance": 0.20,
    "factual_correctness": 0.25,
    "completeness": 0.20,
    "contextual_faithfulness": 0.25,
    "clarity": 0.10,
}

RUBRIC_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "relevance": {
        "description": "Does the answer directly address the question asked?",
        "scale": {
            1: "Completely off-topic or fails to address the question.",
            2: "Tangentially related with major irrelevant content or misses the core intent.",
            3: "Partially addresses the question but drifts or misses key focus.",
            4: "Directly addresses the question with minor extraneous detail.",
            5: "Directly, precisely, and wholly addresses the question.",
        },
    },
    "factual_correctness": {
        "description": (
            "Is the answer factually consistent with the reference answer and established facts "
            "(including appropriate refusal/correction when expected)?"
        ),
        "scale": {
            1: "Completely incorrect, fabricates core facts, or accepts a false premise as true.",
            2: "Contains major factual errors or contradicts key reference facts.",
            3: "Partially correct; contains some accurate points alongside notable inaccuracies or omissions.",
            4: "Mostly accurate and consistent with the reference answer, with trivial imprecision.",
            5: "Fully consistent with the reference answer (including accurate paraphrases or appropriate refusals).",
        },
    },
    "completeness": {
        "description": "Does the answer include all key information required to answer the question?",
        "scale": {
            1: "Missing virtually all required information.",
            2: "Includes only minimal fragments of the needed answer.",
            3: "Covers some key points but omits important conditions, caveats, or parts.",
            4: "Covers nearly all essential information with only minor details omitted.",
            5: "Complete coverage of all key points present in the reference answer.",
        },
    },
    "contextual_faithfulness": {
        "description": (
            "When context is supplied, are all claims in the answer supported by the supplied context "
            "(judged strictly against the supplied context even if counterfactual)? "
            "Return 'NOT_APPLICABLE' when no context is supplied."
        ),
        "scale": {
            1: "Directly contradicts the supplied context or fabricates extensive unsupported claims.",
            2: "Contains major unsupported claims or notable contradiction with the context.",
            3: "Mixes context-supported facts with unsupported extrapolations.",
            4: "Almost entirely supported by context with negligible unsupported phrasing.",
            5: "Every claim is strictly supported by the supplied context, or appropriately declines when unanswerable.",
        },
    },
    "clarity": {
        "description": "Is the answer understandable, coherent, and well-structured?",
        "scale": {
            1: "Incoherent, garbled, or unreadable.",
            2: "Hard to follow with poor structure or confusing wording.",
            3: "Understandable but awkward, wordy, or disorganized.",
            4: "Clear, readable, and reasonably concise.",
            5: "Exceptionally clear, coherent, concise, and well-structured.",
        },
    },
}


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for the LLM-as-a-Judge evaluator."""

    judge_model_key_or_id: str = "qwen2.5-1.5b"
    prompt_version: str = JUDGE_PROMPT_VERSION
    temperature: float = 0.0
    max_new_tokens: int = 512
    timeout_seconds: int = 60
    seed: int = 42
    dimension_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DIMENSION_WEIGHTS)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        return asdict(self)
