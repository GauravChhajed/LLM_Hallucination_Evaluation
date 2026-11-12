"""Unit tests for Phase 5 lexical, semantic similarity, and correctness evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    BaseEmbeddingProvider,
    EmbeddingBackendError,
    EvaluationRecord,
    ReferenceCorrectnessEvaluator,
    ResponseEvaluator,
    SemanticSimilarityEvaluator,
    SentenceTransformerEmbedder,
    compute_exact_match,
    compute_lexical_metrics,
    compute_normalized_exact_match,
    cosine_similarity,
    evaluation_records_to_dataframe,
    load_evaluation_results,
    normalize_text,
    save_evaluation_results,
)
from src.generation import GenerationResult


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic in-memory mock embedder mapping text keywords to 4D vectors."""

    def __init__(self, model_name: str = "mock-all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def _embed_one(self, text: str) -> np.ndarray:
        low = text.lower()
        # High similarity cluster for capital of Australia paraphrases
        if "australia" in low or "canberra" in low:
            return np.array([0.9, 0.3, 0.1, 0.0], dtype=np.float64)
        # High similarity cluster for 17*24 math statements (even if wrong number!)
        if "17" in low and "24" in low:
            return np.array([0.1, 0.9, 0.2, 0.1], dtype=np.float64)
        if "408" in low:
            return np.array([0.1, 0.88, 0.22, 0.1], dtype=np.float64)
        if "409" in low:
            return np.array([0.1, 0.89, 0.21, 0.1], dtype=np.float64)
        # Unrelated topic
        if "banana" in low or "tokyo" in low:
            return np.array([0.0, 0.0, 0.1, 0.95], dtype=np.float64)
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts])


def test_exact_match_and_normalization_preserves_numbers() -> None:
    """Verify exact match, normalized match, and numeric preservation."""
    assert compute_exact_match("Canberra", "Canberra") == 1.0
    assert compute_exact_match("canberra", "Canberra") == 0.0

    assert compute_normalized_exact_match("The capital is Canberra!", "capital is Canberra") == 1.0
    assert normalize_text("Angle is 7.5 degrees.") == "angle is 7.5 degrees"
    assert normalize_text("Probability is 1/6.") == "probability is 1/6"
    assert normalize_text("Population: 1,000 people.") == "population 1000 people"

    # Meaningful numeric differences must NOT be normalized away
    assert compute_normalized_exact_match("7.5 degrees", "75 degrees") == 0.0
    assert compute_normalized_exact_match("1945", "1946") == 0.0


def test_partial_lexical_matches_and_repeated_tokens() -> None:
    """Verify token precision, recall, F1, and repeated-token penalty."""
    partial = compute_lexical_metrics(
        generated="Canberra is the capital of Australia",
        reference="Canberra is Australia's capital city",
    )
    assert 0.0 < partial.token_precision < 1.0
    assert 0.0 < partial.token_recall < 1.0
    assert 0.0 < partial.token_f1 < 1.0

    # Repeated tokens must reduce precision via Counter multiset intersection
    repeated = compute_lexical_metrics(
        generated="Canberra Canberra Canberra",
        reference="Canberra",
    )
    assert repeated.token_recall == pytest.approx(1.0)
    assert repeated.token_precision == pytest.approx(1.0 / 3.0, rel=1e-4)
    assert repeated.token_f1 == pytest.approx(0.5, rel=1e-4)


def test_cosine_similarity_edge_cases() -> None:
    """Verify cosine similarity bounds, zero-norm handling, and dimension errors."""
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)
    assert cosine_similarity(v, -v) == pytest.approx(-1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.zeros(3), v) == 0.0

    with pytest.raises(ValueError):
        cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        cosine_similarity(np.array([]), np.array([]))


def test_semantically_similar_paraphrase_vs_incorrect_numeric_answer() -> None:
    """
    Verify semantic similarity rewards paraphrases while correctness flags wrong numbers
    even when semantic similarity is very high.
    """
    evaluator = ResponseEvaluator(embedder=MockEmbeddingProvider())

    # Paraphrase with different wording but correct meaning
    paraphrase_rec = evaluator.evaluate_single(
        question_id="Q001",
        model_id="mock-model-a",
        generated_answer="Australia's seat of government is Canberra.",
        reference_answer="Canberra is the capital city of Australia.",
        question_type="General Knowledge",
    )
    assert paraphrase_rec.semantic_similarity == pytest.approx(1.0)
    assert paraphrase_rec.exact_match == 0.0
    assert paraphrase_rec.evaluation_status == "evaluated"

    # Incorrect numerical answer with high surface & embedding similarity
    wrong_num_rec = evaluator.evaluate_single(
        question_id="Q031",
        model_id="mock-model-a",
        generated_answer="17 multiplied by 24 equals 409.",
        reference_answer="17 multiplied by 24 equals 408.",
        question_type="Mathematics",
    )
    assert wrong_num_rec.semantic_similarity is not None
    assert wrong_num_rec.semantic_similarity > 0.95
    correctness = wrong_num_rec.evaluation_metadata["correctness"]
    assert correctness["has_numeric_contradiction"] is True
    assert correctness["correctness_label"] == "contradictory_numeric"
    assert correctness["correctness_score"] <= 0.30


def test_empty_generated_and_empty_reference_handling() -> None:
    """Verify graceful handling of empty generated answers and empty references."""
    evaluator = ResponseEvaluator(embedder=MockEmbeddingProvider())

    empty_gen = evaluator.evaluate_single(
        question_id="Q002",
        model_id="mock-model-a",
        generated_answer="   ",
        reference_answer="Mercury is the smallest planet.",
    )
    assert empty_gen.evaluation_status == "empty_generated_answer"
    assert empty_gen.semantic_similarity is None
    assert empty_gen.token_f1 == 0.0
    assert empty_gen.evaluation_metadata["correctness"]["correctness_label"] == "unassessed"

    empty_ref = evaluator.evaluate_single(
        question_id="Q003",
        model_id="mock-model-a",
        generated_answer="Au",
        reference_answer="",
    )
    assert empty_ref.evaluation_status == "empty_reference_answer"
    assert empty_ref.semantic_similarity is None
    assert empty_ref.exact_match == 0.0


def test_missing_embedding_model_error_handling() -> None:
    """Verify informative errors when embedding model fails without downloading weights."""

    def broken_loader(*args, **kwargs):
        raise OSError("Model weights unavailable offline.")

    broken_embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        preferred_device="cpu",
        local_files_only=True,
        model_loader=broken_loader,
    )

    with pytest.raises(EmbeddingBackendError) as exc_info:
        broken_embedder.encode(["test sentence"])
    assert "all-MiniLM-L6-v2" in str(exc_info.value)

    # Evaluator falls back gracefully to lexical evaluation and records semantic error
    evaluator = ResponseEvaluator(embedder=broken_embedder)
    rec = evaluator.evaluate_single(
        question_id="Q001",
        model_id="mock-model-a",
        generated_answer="Canberra",
        reference_answer="Canberra is the capital city of Australia.",
    )
    assert rec.evaluation_status == "evaluated_without_semantic"
    assert rec.semantic_similarity is None
    assert rec.token_precision == pytest.approx(1.0)
    assert rec.evaluation_metadata["semantic_status"] == "error"
    assert "all-MiniLM-L6-v2" in (rec.evaluation_metadata["semantic_error"] or "")


def test_batch_multi_model_evaluation_and_persistence(tmp_path) -> None:
    """Verify Phase 4 GenerationResult integration, multi-model evaluation, and JSON/CSV save."""
    dataset_df = pd.DataFrame(
        [
            {
                "question_id": "Q001",
                "question": "Capital of Australia?",
                "reference_answer": "Canberra is the capital city of Australia.",
                "category": "General Knowledge",
                "difficulty": "easy",
                "expected_behavior": "factual_answer",
                "hallucination_test_type": "factual_precision",
            },
            {
                "question_id": "Q031",
                "question": "What is 17 * 24?",
                "reference_answer": "17 multiplied by 24 equals 408.",
                "category": "Mathematics",
                "difficulty": "easy",
                "expected_behavior": "factual_answer",
                "hallucination_test_type": "factual_precision",
            },
        ]
    )

    gen_results_a = [
        GenerationResult(
            question_id="Q001",
            model_key="model-a",
            model_id="org/model-a",
            backend="huggingface_local",
            question="Capital of Australia?",
            context="",
            prompt_used="...",
            generated_answer="Canberra is the capital city of Australia.",
            generation_time_seconds=0.45,
            status="success",
        ),
        GenerationResult(
            question_id="Q031",
            model_key="model-a",
            model_id="org/model-a",
            backend="huggingface_local",
            question="What is 17 * 24?",
            context="",
            prompt_used="...",
            generated_answer="408",
            generation_time_seconds=0.35,
            status="success",
        ),
    ]
    gen_results_b = [
        GenerationResult(
            question_id="Q001",
            model_key="model-b",
            model_id="org/model-b",
            backend="huggingface_local",
            question="Capital of Australia?",
            context="",
            prompt_used="...",
            generated_answer="Tokyo",
            generation_time_seconds=0.25,
            status="success",
        ),
        GenerationResult(
            question_id="Q031",
            model_key="model-b",
            model_id="org/model-b",
            backend="huggingface_local",
            question="What is 17 * 24?",
            context="",
            prompt_used="...",
            generated_answer="17 multiplied by 24 equals 409.",
            generation_time_seconds=0.30,
            status="success",
        ),
    ]

    evaluator = ResponseEvaluator(embedder=MockEmbeddingProvider())
    multi_eval = evaluator.evaluate_multiple_models(
        {"model-a": gen_results_a, "model-b": gen_results_b},
        dataset_df=dataset_df,
    )
    assert set(multi_eval.keys()) == {"model-a", "model-b"}
    assert len(multi_eval["model-a"]) == 2
    assert len(multi_eval["model-b"]) == 2
    assert multi_eval["model-a"][0].normalized_exact_match == 1.0
    assert multi_eval["model-b"][1].evaluation_metadata["correctness"]["has_numeric_contradiction"] is True

    all_records = multi_eval["model-a"] + multi_eval["model-b"]
    paths = save_evaluation_results(all_records, output_dir=tmp_path, run_name="phase5_eval")
    assert paths["json"].exists() and paths["csv"].exists()

    reloaded = load_evaluation_results(paths["json"])
    assert len(reloaded) == 4
    assert isinstance(reloaded[0], EvaluationRecord)

    df_out = pd.read_csv(paths["csv"])
    assert len(df_out) == 4
    assert set(["question_id", "model_id", "exact_match", "token_f1", "semantic_similarity"]).issubset(
        df_out.columns
    )
