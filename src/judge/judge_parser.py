"""
Strict JSON extraction and schema validation for LLM-as-a-Judge outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.judge.judge_config import DIMENSIONS

NOT_APPLICABLE_LABEL: str = "NOT_APPLICABLE"

REQUIRED_SCORE_KEYS: Dict[str, str] = {
    "relevance": "relevance_score",
    "factual_correctness": "factual_correctness_score",
    "completeness": "completeness_score",
    "contextual_faithfulness": "contextual_faithfulness_score",
    "clarity": "clarity_score",
}


@dataclass
class JudgeParseResult:
    """Validated structured fields parsed from raw judge output."""

    is_valid: bool
    relevance_score: Optional[int] = None
    factual_correctness_score: Optional[int] = None
    completeness_score: Optional[int] = None
    contextual_faithfulness_score: Optional[Union[int, str]] = None
    clarity_score: Optional[int] = None
    dimension_explanations: Dict[str, str] = field(default_factory=dict)
    judge_confidence: Optional[float] = None
    overall_summary: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from raw LLM output (plain JSON or fenced markdown block).

    Raises ValueError if no valid JSON object can be parsed.
    """
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Judge returned empty output instead of JSON.")

    # Try direct JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try fenced ```json ... ``` blocks
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Try balanced outermost braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in judge output: {exc}") from exc

    raise ValueError("No valid JSON object found in judge output.")


def _validate_int_score(value: Any, field_name: str, errors: List[str]) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"Field '{field_name}' must be an integer in [1, 5], got {value!r}.")
        return None
    if value < 1 or value > 5:
        errors.append(f"Field '{field_name}' out of range [1, 5]: {value}.")
        return None
    return int(value)


def parse_and_validate_judge_output(raw_text: str, has_context: bool) -> JudgeParseResult:
    """
    Parse and strictly validate judge JSON scores, explanations, and confidence.
    """
    try:
        payload = extract_json_object(raw_text)
    except ValueError as exc:
        return JudgeParseResult(is_valid=False, errors=[str(exc)])

    errors: List[str] = []
    warnings: List[str] = []

    missing_keys = [
        k for k in list(REQUIRED_SCORE_KEYS.values()) + ["dimension_explanations", "judge_confidence"]
        if k not in payload
    ]
    if missing_keys:
        errors.append(f"Missing required JSON fields: {', '.join(missing_keys)}")

    rel_score = _validate_int_score(payload.get("relevance_score"), "relevance_score", errors) if "relevance_score" in payload else None
    fact_score = _validate_int_score(payload.get("factual_correctness_score"), "factual_correctness_score", errors) if "factual_correctness_score" in payload else None
    comp_score = _validate_int_score(payload.get("completeness_score"), "completeness_score", errors) if "completeness_score" in payload else None
    clarity_score = _validate_int_score(payload.get("clarity_score"), "clarity_score", errors) if "clarity_score" in payload else None

    ctx_raw = payload.get("contextual_faithfulness_score")
    ctx_score: Optional[Union[int, str]] = None
    if "contextual_faithfulness_score" in payload:
        if not has_context:
            if isinstance(ctx_raw, str) and ctx_raw.strip().upper() == NOT_APPLICABLE_LABEL:
                ctx_score = NOT_APPLICABLE_LABEL
            else:
                ctx_score = NOT_APPLICABLE_LABEL
                warnings.append(
                    f"No context was supplied; coerced contextual_faithfulness_score={ctx_raw!r} to '{NOT_APPLICABLE_LABEL}'."
                )
        else:
            if isinstance(ctx_raw, str) and ctx_raw.strip().upper() == NOT_APPLICABLE_LABEL:
                errors.append(
                    "contextual_faithfulness_score cannot be 'NOT_APPLICABLE' when context was provided."
                )
            else:
                ctx_score = _validate_int_score(
                    ctx_raw, "contextual_faithfulness_score", errors
                )

    explanations_raw = payload.get("dimension_explanations")
    explanations: Dict[str, str] = {}
    if isinstance(explanations_raw, dict):
        for dim in DIMENSIONS:
            exp_val = str(explanations_raw.get(dim, "") or "").strip()
            explanations[dim] = exp_val
            if not exp_val:
                warnings.append(f"Missing explanation for dimension '{dim}'.")
    elif "dimension_explanations" in payload:
        errors.append("Field 'dimension_explanations' must be a JSON object.")

    conf_raw = payload.get("judge_confidence")
    judge_conf: Optional[float] = None
    if "judge_confidence" in payload:
        if isinstance(conf_raw, bool) or not isinstance(conf_raw, (int, float)):
            errors.append(f"Field 'judge_confidence' must be a float in [0.0, 1.0], got {conf_raw!r}.")
        else:
            conf_f = float(conf_raw)
            if conf_f < 0.0 or conf_f > 1.0:
                errors.append(f"Field 'judge_confidence' out of range [0.0, 1.0]: {conf_f}.")
            else:
                judge_conf = round(conf_f, 4)

    summary = str(payload.get("overall_summary", "") or "").strip()

    return JudgeParseResult(
        is_valid=len(errors) == 0,
        relevance_score=rel_score,
        factual_correctness_score=fact_score,
        completeness_score=comp_score,
        contextual_faithfulness_score=ctx_score,
        clarity_score=clarity_score,
        dimension_explanations=explanations,
        judge_confidence=judge_conf,
        overall_summary=summary,
        errors=errors,
        warnings=warnings,
    )
