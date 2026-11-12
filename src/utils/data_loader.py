"""
Dataset loading, schema validation, and summary analytics utilities.

Supports CSV and JSON evaluation datasets, validates required columns,
detects missing values, duplicate IDs/questions, invalid categories or
difficulties, and computes category distributions and summary statistics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

CORE_COLUMNS: Tuple[str, ...] = (
    "question_id",
    "question",
    "context",
    "reference_answer",
    "category",
    "difficulty",
)

EXTENDED_COLUMNS: Tuple[str, ...] = (
    "expected_behavior",
    "hallucination_test_type",
)

OPTIONAL_COLUMNS: Tuple[str, ...] = (
    "reference_source",
)

REQUIRED_COLUMNS: Tuple[str, ...] = CORE_COLUMNS + EXTENDED_COLUMNS

VALID_CATEGORIES: Tuple[str, ...] = (
    "General Knowledge",
    "Science and Technology",
    "Mathematics",
    "History",
    "Logical Reasoning",
    "Context-based Question Answering",
    "Adversarial and Hallucination Testing",
)

VALID_DIFFICULTIES: Tuple[str, ...] = ("easy", "medium", "hard")

NON_EMPTY_FIELDS: Tuple[str, ...] = (
    "question_id",
    "question",
    "reference_answer",
    "category",
    "difficulty",
    "expected_behavior",
    "hallucination_test_type",
)


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset fails schema or content validation."""

    def __init__(self, message: str, result: Optional["DatasetValidationResult"] = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class DatasetValidationResult:
    """Structured report produced by `validate_dataset`."""

    is_valid: bool
    num_rows: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    duplicate_ids: List[str] = field(default_factory=list)
    duplicate_questions: List[str] = field(default_factory=list)
    empty_questions: List[str] = field(default_factory=list)
    missing_reference_answers: List[str] = field(default_factory=list)
    missing_fields_by_column: Dict[str, int] = field(default_factory=dict)
    invalid_categories: List[str] = field(default_factory=list)
    invalid_difficulties: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the validation result to a plain dictionary."""
        return asdict(self)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize string columns while preserving missing values as empty strings where appropriate."""
    normalized = df.copy()
    for col in normalized.columns:
        normalized[col] = normalized[col].fillna("").astype(str).str.strip()
    return normalized


def load_csv_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """Load a dataset from a CSV file without turning empty context strings into NaN."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset CSV file not found: {file_path}")
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    return _normalize_dataframe(df)


def load_json_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a dataset from a JSON file.

    Accepts either a top-level list of record objects or a dictionary containing
    a `"questions"` or `"data"` list.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict):
        if "questions" in payload and isinstance(payload["questions"], list):
            records = payload["questions"]
        elif "data" in payload and isinstance(payload["data"], list):
            records = payload["data"]
        else:
            raise ValueError(
                "JSON object dataset must contain a 'questions' or 'data' list."
            )
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("JSON dataset must be a list of objects or a dict with 'questions'/'data'.")

    df = pd.DataFrame(records)
    return _normalize_dataframe(df)


def validate_dataset(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
    allowed_categories: Optional[Sequence[str]] = VALID_CATEGORIES,
    allowed_difficulties: Optional[Sequence[str]] = VALID_DIFFICULTIES,
    require_context_for_context_qa: bool = True,
) -> DatasetValidationResult:
    """
    Validate dataset schema, completeness, uniqueness, and categorical integrity.

    Returns a DatasetValidationResult describing all detected errors and warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        return DatasetValidationResult(
            is_valid=False,
            num_rows=0,
            errors=["Dataset is empty (0 rows)."],
        )

    work_df = _normalize_dataframe(df)
    missing_cols = [col for col in required_columns if col not in work_df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # Missing-value detection across non-empty required fields
    missing_fields_by_column: Dict[str, int] = {}
    empty_questions: List[str] = []
    missing_reference_answers: List[str] = []

    for col in required_columns:
        if col not in work_df.columns:
            continue
        if col == "context":
            continue  # Empty context is valid for non-context QA items
        empty_mask = work_df[col].str.len() == 0
        empty_count = int(empty_mask.sum())
        if empty_count > 0:
            missing_fields_by_column[col] = empty_count
            if col == "question":
                empty_questions = (
                    work_df.loc[empty_mask, "question_id"].tolist()
                    if "question_id" in work_df.columns
                    else [f"row_{idx}" for idx in work_df.index[empty_mask]]
                )
            elif col == "reference_answer":
                missing_reference_answers = (
                    work_df.loc[empty_mask, "question_id"].tolist()
                    if "question_id" in work_df.columns
                    else [f"row_{idx}" for idx in work_df.index[empty_mask]]
                )
            errors.append(f"Column '{col}' has {empty_count} missing/empty value(s).")

    # Duplicate question_id detection
    duplicate_ids: List[str] = []
    if "question_id" in work_df.columns:
        non_empty_ids = work_df["question_id"][work_df["question_id"].str.len() > 0]
        dup_id_mask = non_empty_ids.duplicated(keep=False)
        if dup_id_mask.any():
            duplicate_ids = sorted(non_empty_ids[dup_id_mask].unique().tolist())
            errors.append(f"Duplicate question_id values found: {', '.join(duplicate_ids)}")

    # Duplicate question text detection (case-insensitive)
    duplicate_questions: List[str] = []
    if "question" in work_df.columns:
        norm_q = work_df["question"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        non_empty_q = norm_q[norm_q.str.len() > 0]
        dup_q_mask = non_empty_q.duplicated(keep=False)
        if dup_q_mask.any():
            duplicate_questions = sorted(work_df.loc[non_empty_q[dup_q_mask].index, "question"].unique().tolist())
            errors.append(f"Duplicate question texts detected ({len(duplicate_questions)} unique duplicates).")

    # Category validation
    invalid_categories: List[str] = []
    if allowed_categories is not None and "category" in work_df.columns:
        allowed_cat_set = set(allowed_categories)
        present_cats = [c for c in work_df["category"].unique().tolist() if c]
        invalid_categories = sorted([c for c in present_cats if c not in allowed_cat_set])
        if invalid_categories:
            errors.append(
                f"Invalid categories detected: {', '.join(invalid_categories)}. "
                f"Allowed: {', '.join(allowed_categories)}"
            )

    # Difficulty validation
    invalid_difficulties: List[str] = []
    if allowed_difficulties is not None and "difficulty" in work_df.columns:
        allowed_diff_set = {d.lower() for d in allowed_difficulties}
        present_diffs = [d for d in work_df["difficulty"].unique().tolist() if d]
        invalid_difficulties = sorted([d for d in present_diffs if d.lower() not in allowed_diff_set])
        if invalid_difficulties:
            errors.append(
                f"Invalid difficulty levels detected: {', '.join(invalid_difficulties)}. "
                f"Allowed: {', '.join(allowed_difficulties)}"
            )

    # Context-based QA must have non-empty context
    if (
        require_context_for_context_qa
        and "category" in work_df.columns
        and "context" in work_df.columns
    ):
        ctx_qa_mask = work_df["category"] == "Context-based Question Answering"
        missing_ctx_mask = ctx_qa_mask & (work_df["context"].str.len() == 0)
        if missing_ctx_mask.any():
            missing_ids = work_df.loc[missing_ctx_mask, "question_id"].tolist()
            errors.append(
                "Context-based Question Answering entries missing context: "
                + ", ".join(missing_ids)
            )

    return DatasetValidationResult(
        is_valid=len(errors) == 0,
        num_rows=len(work_df),
        errors=errors,
        warnings=warnings,
        missing_columns=missing_cols,
        duplicate_ids=duplicate_ids,
        duplicate_questions=duplicate_questions,
        empty_questions=empty_questions,
        missing_reference_answers=missing_reference_answers,
        missing_fields_by_column=missing_fields_by_column,
        invalid_categories=invalid_categories,
        invalid_difficulties=invalid_difficulties,
    )


def load_dataset(
    path: Union[str, Path],
    validate: bool = True,
    raise_on_error: bool = True,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
    allowed_categories: Optional[Sequence[str]] = VALID_CATEGORIES,
) -> Tuple[pd.DataFrame, DatasetValidationResult]:
    """
    Load a CSV or JSON evaluation dataset and optionally validate it.

    Returns (dataframe, validation_result). Raises DatasetValidationError if
    `validate=True`, `raise_on_error=True`, and validation fails.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = load_csv_dataset(file_path)
    elif suffix == ".json":
        df = load_json_dataset(file_path)
    else:
        raise ValueError(f"Unsupported dataset format '{suffix}'. Expected '.csv' or '.json'.")

    if validate:
        result = validate_dataset(
            df,
            required_columns=required_columns,
            allowed_categories=allowed_categories,
        )
        if not result.is_valid and raise_on_error:
            raise DatasetValidationError(
                "Dataset validation failed:\n- " + "\n- ".join(result.errors),
                result=result,
            )
    else:
        result = DatasetValidationResult(is_valid=True, num_rows=len(df))

    return df, result


def get_category_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Return a deterministic mapping of category name to question count."""
    if df.empty or "category" not in df.columns:
        return {}
    counts = df["category"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive descriptive statistics for an evaluation dataset."""
    if df.empty:
        return {
            "total_questions": 0,
            "categories": {},
            "difficulties": {},
            "hallucination_test_types": {},
            "expected_behaviors": {},
            "questions_with_context": 0,
            "questions_without_context": 0,
            "avg_question_words": 0.0,
            "avg_reference_words": 0.0,
        }

    work_df = _normalize_dataframe(df)
    has_ctx = (
        work_df["context"].str.len() > 0
        if "context" in work_df.columns
        else pd.Series([False] * len(work_df))
    )

    q_words = (
        work_df["question"].str.split().str.len().mean()
        if "question" in work_df.columns
        else 0.0
    )
    r_words = (
        work_df["reference_answer"].str.split().str.len().mean()
        if "reference_answer" in work_df.columns
        else 0.0
    )

    has_source = (
        work_df["reference_source"].str.len() > 0
        if "reference_source" in work_df.columns
        else pd.Series([False] * len(work_df))
    )

    return {
        "total_questions": int(len(work_df)),
        "categories": get_category_distribution(work_df),
        "difficulties": {
            str(k): int(v)
            for k, v in work_df["difficulty"].str.lower().value_counts().to_dict().items()
        }
        if "difficulty" in work_df.columns
        else {},
        "hallucination_test_types": {
            str(k): int(v)
            for k, v in work_df["hallucination_test_type"].value_counts().to_dict().items()
        }
        if "hallucination_test_type" in work_df.columns
        else {},
        "expected_behaviors": {
            str(k): int(v)
            for k, v in work_df["expected_behavior"].value_counts().to_dict().items()
        }
        if "expected_behavior" in work_df.columns
        else {},
        "questions_with_context": int(has_ctx.sum()),
        "questions_without_context": int((~has_ctx).sum()),
        "questions_with_reference_source": int(has_source.sum()),
        "avg_question_words": round(float(q_words), 2),
        "avg_reference_words": round(float(r_words), 2),
    }


def save_dataset_outputs(
    df: pd.DataFrame,
    raw_json_path: Union[str, Path],
    processed_json_path: Union[str, Path],
    csv_path: Union[str, Path],
) -> None:
    """
    Validate and export the canonical dataset to raw JSON, processed JSON, and CSV.
    """
    result = validate_dataset(df)
    if not result.is_valid:
        raise DatasetValidationError(
            "Cannot export invalid dataset:\n- " + "\n- ".join(result.errors),
            result=result,
        )

    work_df = _normalize_dataframe(df)
    records = work_df.to_dict(orient="records")
    summary = get_dataset_summary(work_df)

    for target in (raw_json_path, processed_json_path, csv_path):
        Path(target).parent.mkdir(parents=True, exist_ok=True)

    with Path(raw_json_path).open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    processed_payload = {
        "metadata": {
            "schema_version": "1.0.0",
            "validation_passed": result.is_valid,
            "summary": summary,
        },
        "questions": records,
    }
    with Path(processed_json_path).open("w", encoding="utf-8") as fh:
        json.dump(processed_payload, fh, indent=2, ensure_ascii=False)

    work_df.to_csv(csv_path, index=False, encoding="utf-8")
