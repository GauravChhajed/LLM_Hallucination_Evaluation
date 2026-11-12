"""
Batch response generation and persistence coordinator.

Runs identical questions and contexts through selected models and stores
structured model response logs under results/model_responses/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.generation.generator import BaseLLMGenerator, GenerationResult


def generate_responses_for_dataset(
    df: pd.DataFrame,
    generator: BaseLLMGenerator,
    max_questions: Optional[int] = None,
    system_prompt: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, GenerationResult], None]] = None,
) -> List[GenerationResult]:
    """
    Generate responses for every row (or a deterministic subset) of a dataset.

    Preserves dataset order so Model A and Model B receive identical inputs.
    """
    subset = df.head(max_questions) if max_questions is not None else df
    total = len(subset)
    results: List[GenerationResult] = []

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        qid = str(row["question_id"])
        question = str(row["question"])
        context = str(row.get("context", "") or "")

        result = generator.generate(
            question_id=qid,
            question=question,
            context=context,
            system_prompt=system_prompt,
        )
        results.append(result)
        if progress_callback is not None:
            progress_callback(idx, total, result)

    return results


def generate_paired_responses(
    df: pd.DataFrame,
    generator_a: BaseLLMGenerator,
    generator_b: BaseLLMGenerator,
    max_questions: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> Tuple[List[GenerationResult], List[GenerationResult]]:
    """
    Run paired generation for Model A and Model B over identical questions/contexts.
    """
    results_a = generate_responses_for_dataset(
        df=df,
        generator=generator_a,
        max_questions=max_questions,
        system_prompt=system_prompt,
    )
    results_b = generate_responses_for_dataset(
        df=df,
        generator=generator_b,
        max_questions=max_questions,
        system_prompt=system_prompt,
    )
    return results_a, results_b


def results_to_dataframe(results: Sequence[GenerationResult]) -> pd.DataFrame:
    """Convert a sequence of GenerationResult records into a tabular DataFrame."""
    rows = []
    for item in results:
        row = item.to_dict()
        row["metadata"] = json.dumps(row.get("metadata", {}), ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def save_generation_results(
    results: Sequence[GenerationResult],
    output_dir: Path,
    run_name: str,
) -> Dict[str, Path]:
    """Persist generation results to both JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}.csv"

    payload = [item.to_dict() for item in results]
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    df = results_to_dataframe(results)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    return {"json": json_path, "csv": csv_path}


def load_generation_results(json_path: Path) -> List[GenerationResult]:
    """Reload GenerationResult records from a previously saved JSON file."""
    with json_path.open("r", encoding="utf-8") as fh:
        payload: List[Dict[str, Any]] = json.load(fh)
    return [GenerationResult(**entry) for entry in payload]
