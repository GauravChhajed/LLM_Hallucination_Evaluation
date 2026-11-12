"""
Prompt construction utilities for fair, context-aware LLM comparison.

Ensures every evaluated model receives an identical, neutral instruction and
standardized formatting, while allowing alternate templates in future experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_SYSTEM_PROMPT: str = (
    "You are answering questions in an evaluation setting. "
    "Answer the question using the information available to you. "
    "If context is provided, consider it when formulating your answer. "
    "Provide a concise answer."
)


@dataclass(frozen=True)
class PromptTemplate:
    """Configurable prompt template for standardized or experimental runs."""

    name: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    context_format: str = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    no_context_format: str = "Question:\n{question}\n\nAnswer:"


STANDARD_EVAL_TEMPLATE = PromptTemplate(name="standard_neutral_v1")


@dataclass(frozen=True)
class PromptPayload:
    """Normalized prompt representation supporting both chat and plain-text models."""

    question: str
    context: str
    system_prompt: str
    user_prompt: str
    plain_text_prompt: str
    messages: List[Dict[str, str]]
    has_context: bool
    template_name: str = STANDARD_EVAL_TEMPLATE.name


def build_qa_prompt(
    question: str,
    context: str = "",
    system_prompt: Optional[str] = None,
    template: Optional[PromptTemplate] = None,
) -> PromptPayload:
    """
    Build a standardized prompt payload from a question and optional context.

    Uses STANDARD_EVAL_TEMPLATE by default so all compared models receive the
    same neutral instructions without benchmark-specific coaching.
    Raises ValueError if the question is empty or whitespace-only.
    """
    clean_question = (question or "").strip()
    if not clean_question:
        raise ValueError("Cannot build a prompt for an empty question.")

    active_template = template or STANDARD_EVAL_TEMPLATE
    clean_context = (context or "").strip()
    sys_msg = (system_prompt or active_template.system_prompt).strip()
    has_context = len(clean_context) > 0

    if has_context:
        user_prompt = active_template.context_format.format(
            context=clean_context,
            question=clean_question,
        )
    else:
        user_prompt = active_template.no_context_format.format(
            question=clean_question,
        )

    plain_text_prompt = f"{sys_msg}\n\n{user_prompt}"
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_prompt},
    ]

    return PromptPayload(
        question=clean_question,
        context=clean_context,
        system_prompt=sys_msg,
        user_prompt=user_prompt,
        plain_text_prompt=plain_text_prompt,
        messages=messages,
        has_context=has_context,
        template_name=active_template.name,
    )


def format_prompt_for_tokenizer(
    payload: PromptPayload,
    tokenizer: Any,
    architecture: str = "causal_lm",
) -> str:
    """
    Format a PromptPayload into the exact string fed to a Hugging Face tokenizer.

    Uses the model's native chat template for causal instruction/chat models when
    available, and falls back to plain-text formatting for seq2seq or base models.
    """
    if architecture == "seq2seq_lm":
        return payload.plain_text_prompt

    chat_template = getattr(tokenizer, "chat_template", None)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if chat_template and callable(apply_template):
        try:
            rendered = apply_template(
                payload.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            # Fallback if a custom tokenizer rejects system-role messages
            merged_messages = [
                {
                    "role": "user",
                    "content": f"{payload.system_prompt}\n\n{payload.user_prompt}",
                }
            ]
            try:
                rendered = apply_template(
                    merged_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception:
                pass

    return payload.plain_text_prompt
