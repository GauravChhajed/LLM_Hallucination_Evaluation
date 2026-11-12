"""Unit tests for dataset loading, schema validation, and benchmark integrity."""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
import math

import pandas as pd
import pytest

from src.utils.config import get_config
from src.utils.data_loader import (
    OPTIONAL_COLUMNS,
    DatasetValidationError,
    get_category_distribution,
    get_dataset_summary,
    load_dataset,
    validate_dataset,
)

EXPECTED_CATEGORY_COUNTS = {
    "General Knowledge": 15,
    "Science and Technology": 15,
    "Mathematics": 15,
    "History": 10,
    "Logical Reasoning": 10,
    "Context-based Question Answering": 20,
    "Adversarial and Hallucination Testing": 15,
}


def test_canonical_dataset_loading_and_distribution() -> None:
    """Verify CSV, raw JSON, and processed JSON match record-for-record with expected counts."""
    cfg = get_config()
    csv_path = cfg.paths.default_dataset_path
    raw_json = cfg.paths.raw_data_dir / "original_dataset.json"
    processed_json = cfg.paths.processed_data_dir / "validated_dataset.json"

    df_csv, val_csv = load_dataset(csv_path)
    df_raw, val_raw = load_dataset(raw_json)
    df_proc, val_proc = load_dataset(processed_json)

    assert val_csv.is_valid and val_raw.is_valid and val_proc.is_valid
    assert len(df_csv) == 100
    assert df_csv["question_id"].nunique() == 100
    assert df_csv.equals(df_raw[df_csv.columns])
    assert df_csv.equals(df_proc[df_csv.columns])

    dist = get_category_distribution(df_csv)
    assert dist == EXPECTED_CATEGORY_COUNTS

    summary = get_dataset_summary(df_csv)
    assert summary["total_questions"] == 100
    assert summary["questions_with_context"] == 20
    assert summary["questions_without_context"] == 80
    assert summary["questions_with_reference_source"] == 91


def test_q064_unique_solution_and_q067_distinction() -> None:
    """Verify Q064 has a unique permutation solution and Q067 separates cause from action."""
    cfg = get_config()
    df, _ = load_dataset(cfg.paths.default_dataset_path)

    # Brute-force verify Q064 constraint uniqueness
    colors = ["Red", "Blue", "Green", "Yellow", "White"]
    solutions = []
    for perm in permutations(colors):
        pos = {c: idx + 1 for idx, c in enumerate(perm)}
        if (
            pos["Blue"] == pos["Red"] + 1
            and pos["Green"] == 5
            and pos["Yellow"] == pos["Blue"] + 1
            and pos["White"] > pos["Yellow"]
        ):
            solutions.append(perm)

    assert len(solutions) == 1
    assert solutions[0] == ("Red", "Blue", "Yellow", "White", "Green")
    assert solutions[0][1] == "Blue"

    q064 = df.loc[df["question_id"] == "Q064"].iloc[0]
    assert "Yellow is immediately to the right of Blue" in q064["question"]
    assert "White is to the right of Yellow" in q064["question"]
    assert "Blue is at position 2" in q064["reference_answer"]

    q067 = df.loc[df["question_id"] == "Q067"].iloc[0]
    assert "does not state whether the worn gasket caused the pressure drop" in q067["reference_answer"]


def test_math_ground_truths_verified() -> None:
    """Independently check all 15 Mathematics reference answers in the dataset."""
    cfg = get_config()
    df, _ = load_dataset(cfg.paths.default_dataset_path)
    math_df = df.loc[df["category"] == "Mathematics"].set_index("question_id")
    assert len(math_df) == 15

    expected_tokens = {
        "Q031": str(17 * 24),
        "Q032": str((22 - 7) // 3),
        "Q033": str(int(0.5 * 12 * 9)),
        "Q034": str(int(0.15 * 240)),
        "Q035": str(math.gcd(48, 180)),
        "Q036": "12x^2 - 6",
        "Q037": str((6 - 2) * 180),
        "Q038": str(Fraction(sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == 7), 36)),
        "Q039": str(5 * 4 - 3 * 2),
        "Q040": str(2 ** 6),
        "Q041": str(4 ** 3 - 1 ** 3),
        "Q042": str(math.factorial(5) // (math.factorial(2) * math.factorial(2))),
        "Q043": str(math.isqrt(9 ** 2 + 40 ** 2)),
        "Q044": str(Fraction(12, 1) / (1 - Fraction(1, 3))),
        "Q045": str(pow(7, 10, 10)),
    }
    for qid, token in expected_tokens.items():
        assert token in math_df.loc[qid, "reference_answer"], f"{qid} missing {token}"


def test_reference_source_is_optional_on_custom_datasets() -> None:
    """Verify datasets omitting optional reference_source still pass validation."""
    assert "reference_source" in OPTIONAL_COLUMNS
    df = pd.DataFrame(
        [
            {
                "question_id": "C001",
                "question": "What is the boiling point of water at 1 atm?",
                "context": "",
                "reference_answer": "100 degrees Celsius.",
                "category": "Science and Technology",
                "difficulty": "easy",
                "expected_behavior": "factual_answer",
                "hallucination_test_type": "factual_precision",
            }
        ]
    )
    result = validate_dataset(df)
    assert result.is_valid


def test_missing_required_columns(tmp_path) -> None:
    """Verify validation fails when any required column is absent."""
    bad_df = pd.DataFrame(
        [
            {
                "question_id": "Q1",
                "question": "What is 2+2?",
                "context": "",
            }
        ]
    )
    csv_file = tmp_path / "missing_cols.csv"
    bad_df.to_csv(csv_file, index=False)

    with pytest.raises(DatasetValidationError) as exc_info:
        load_dataset(csv_file)

    assert exc_info.value.result is not None
    assert "reference_answer" in exc_info.value.result.missing_columns


def test_duplicate_question_ids() -> None:
    """Verify duplicate question_id entries are detected."""
    df = pd.DataFrame(
        [
            {
                "question_id": "Q001",
                "question": "First question?",
                "context": "",
                "reference_answer": "First answer.",
                "category": "General Knowledge",
                "difficulty": "easy",
                "expected_behavior": "factual_answer",
                "hallucination_test_type": "factual_precision",
            },
            {
                "question_id": "Q001",
                "question": "Second distinct question?",
                "context": "",
                "reference_answer": "Second answer.",
                "category": "General Knowledge",
                "difficulty": "easy",
                "expected_behavior": "factual_answer",
                "hallucination_test_type": "factual_precision",
            },
        ]
    )
    result = validate_dataset(df)
    assert not result.is_valid
    assert "Q001" in result.duplicate_ids


def test_empty_questions() -> None:
    """Verify empty or whitespace-only questions fail validation."""
    df = pd.DataFrame(
        [
            {
                "question_id": "Q001",
                "question": "   ",
                "context": "",
                "reference_answer": "Some answer.",
                "category": "General Knowledge",
                "difficulty": "easy",
                "expected_behavior": "factual_answer",
                "hallucination_test_type": "factual_precision",
            }
        ]
    )
    result = validate_dataset(df)
    assert not result.is_valid
    assert "Q001" in result.empty_questions


def test_missing_reference_answers() -> None:
    """Verify missing or empty reference_answer entries fail validation."""
    df = pd.DataFrame(
        [
            {
                "question_id": "Q002",
                "question": "What is the capital of France?",
                "context": "",
                "reference_answer": "",
                "category": "General Knowledge",
                "difficulty": "easy",
                "expected_behavior": "factual_answer",
                "hallucination_test_type": "factual_precision",
            }
        ]
    )
    result = validate_dataset(df)
    assert not result.is_valid
    assert "Q002" in result.missing_reference_answers


def test_invalid_categories_and_missing_context_for_context_qa() -> None:
    """Verify unknown categories and Context-QA items without context are rejected."""
    df = pd.DataFrame(
        [
            {
                "question_id": "Q010",
                "question": "Invalid category question?",
                "context": "",
                "reference_answer": "Answer.",
                "category": "Quantum Cooking",
                "difficulty": "easy",
                "expected_behavior": "factual_answer",
                "hallucination_test_type": "factual_precision",
            },
            {
                "question_id": "Q011",
                "question": "Context QA without context?",
                "context": "",
                "reference_answer": "Answer.",
                "category": "Context-based Question Answering",
                "difficulty": "medium",
                "expected_behavior": "context_grounded_answer",
                "hallucination_test_type": "context_supported_precision",
            },
        ]
    )
    result = validate_dataset(df)
    assert not result.is_valid
    assert "Quantum Cooking" in result.invalid_categories
    assert any("Q011" in err for err in result.errors)
