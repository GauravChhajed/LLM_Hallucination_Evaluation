"""
Neutral, blind, versioned prompt builder for LLM-as-a-Judge evaluation.

Never discloses generator model identity, model ranking, or peer scores
to the judge model. Enforces identical rubric instructions across all models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.judge.judge_config import JUDGE_PROMPT_VERSION, RUBRIC_DEFINITIONS

JUDGE_SYSTEM_PROMPT: str = (
    "You are an impartial, rigorous evaluation judge assessing an AI assistant's answer. "
    "Evaluate strictly according to the provided rubric. "
    "Do not reward verbosity over accuracy, and do not penalize valid paraphrases that convey "
    "the same factual meaning as the reference answer. "
    "If a question cannot be answered from the supplied context, contains a false premise, or "
    "cannot be known with certainty, an honest refusal or premise correction is the correct behavior. "
    "If a Context is provided, judge contextual_faithfulness strictly against that Context—even if "
    "the Context describes a hypothetical or counterfactual scenario. "
    "If no Context is provided, set contextual_faithfulness_score to \"NOT_APPLICABLE\". "
    "Return ONLY valid JSON matching the requested schema."
)


@dataclass(frozen=True)
class JudgePromptPayload:
    """Structured prompt sent to a judge backend."""

    question_id: str
    prompt_version: str
    has_context: bool
    is_counterfactual_context: bool
    system_prompt: str
    user_prompt: str
    messages: List[Dict[str, str]]
    plain_text_prompt: str


def _format_rubric_text() -> str:
    lines: List[str] = []
    for dim, info in RUBRIC_DEFINITIONS.items():
        lines.append(f"- {dim}: {info['description']}")
        for score, desc in info["scale"].items():
            lines.append(f"  {score}: {desc}")
    return "\n".join(lines)


def build_judge_prompt(
    question_id: str,
    question: str,
    reference_answer: str,
    generated_answer: str,
    context: str = "",
    is_counterfactual_context: bool = False,
    prompt_version: str = JUDGE_PROMPT_VERSION,
) -> JudgePromptPayload:
    """
    Build a blind, rubric-standardized judge prompt without leaking model identity.
    """
    clean_question = (question or "").strip()
    clean_reference = (reference_answer or "").strip()
    clean_answer = (generated_answer or "").strip()
    clean_context = (context or "").strip()
    has_context = len(clean_context) > 0

    context_block = clean_context if has_context else "[NO CONTEXT SUPPLIED]"
    cf_note = (
        "\nNote: The supplied Context may intentionally present hypothetical or counterfactual "
        "assumptions. Evaluate contextual_faithfulness strictly relative to the supplied Context."
        if (has_context and is_counterfactual_context)
        else ""
    )
    faith_instruction = (
        'Score integer 1-5 strictly against the supplied Context.'
        if has_context
        else 'MUST be the exact string "NOT_APPLICABLE" because no context was supplied.'
    )

    user_prompt = (
        f"Prompt Version: {prompt_version}\n"
        f"Question ID: {question_id}\n\n"
        f"Question:\n{clean_question}\n\n"
        f"Context:\n{context_block}{cf_note}\n\n"
        f"Reference Answer:\n{clean_reference}\n\n"
        f"Candidate Answer (Model Identity Hidden):\n{clean_answer}\n\n"
        f"Evaluation Rubric (1-5 scale):\n{_format_rubric_text()}\n\n"
        "Instructions:\n"
        f"- contextual_faithfulness_score: {faith_instruction}\n"
        "- judge_confidence: Your self-reported confidence (float from 0.0 to 1.0) in your evaluation "
        "(not a calibrated probability or candidate answer score).\n"
        "- Output ONLY valid JSON with these exact keys:\n"
        "{\n"
        '  "relevance_score": <int 1-5>,\n'
        '  "factual_correctness_score": <int 1-5>,\n'
        '  "completeness_score": <int 1-5>,\n'
        '  "contextual_faithfulness_score": <int 1-5 or "NOT_APPLICABLE">,\n'
        '  "clarity_score": <int 1-5>,\n'
        '  "dimension_explanations": {\n'
        '    "relevance": "<short explanation>",\n'
        '    "factual_correctness": "<short explanation>",\n'
        '    "completeness": "<short explanation>",\n'
        '    "contextual_faithfulness": "<short explanation>",\n'
        '    "clarity": "<short explanation>"\n'
        "  },\n"
        '  "judge_confidence": <float 0.0-1.0>,\n'
        '  "overall_summary": "<brief justification>"\n'
        "}"
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    plain_text_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"

    return JudgePromptPayload(
        question_id=question_id,
        prompt_version=prompt_version,
        has_context=has_context,
        is_counterfactual_context=is_counterfactual_context,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        messages=messages,
        plain_text_prompt=plain_text_prompt,
    )
