"""Reference-based lexical, semantic, and correctness evaluation package."""

from src.evaluation.correctness import (
    CorrectnessAssessment,
    CorrectnessConfig,
    ReferenceCorrectnessEvaluator,
    detects_refusal_or_correction,
    extract_key_terms,
    extract_numeric_tokens,
    is_refusal_or_correction_expected,
)
from src.evaluation.evaluator import (
    EvaluationRecord,
    ResponseEvaluator,
    evaluation_records_to_dataframe,
    load_evaluation_results,
    save_evaluation_results,
)
from src.evaluation.lexical_metrics import (
    LexicalMetricsResult,
    compute_exact_match,
    compute_lexical_metrics,
    compute_normalized_exact_match,
    compute_token_f1_from_tokens,
    normalize_text,
    tokenize_text,
)
from src.evaluation.semantic_similarity import (
    BaseEmbeddingProvider,
    EmbeddingBackendError,
    SemanticSimilarityEvaluator,
    SemanticSimilarityResult,
    SentenceTransformerEmbedder,
    batch_cosine_similarity,
    cosine_similarity,
)

__all__ = [
    "BaseEmbeddingProvider",
    "CorrectnessAssessment",
    "CorrectnessConfig",
    "EmbeddingBackendError",
    "EvaluationRecord",
    "LexicalMetricsResult",
    "ReferenceCorrectnessEvaluator",
    "ResponseEvaluator",
    "SemanticSimilarityEvaluator",
    "SemanticSimilarityResult",
    "SentenceTransformerEmbedder",
    "batch_cosine_similarity",
    "compute_exact_match",
    "compute_lexical_metrics",
    "compute_normalized_exact_match",
    "compute_token_f1_from_tokens",
    "cosine_similarity",
    "detects_refusal_or_correction",
    "evaluation_records_to_dataframe",
    "extract_key_terms",
    "extract_numeric_tokens",
    "is_refusal_or_correction_expected",
    "load_evaluation_results",
    "normalize_text",
    "save_evaluation_results",
    "tokenize_text",
]
