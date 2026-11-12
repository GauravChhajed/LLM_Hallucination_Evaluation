"""
Semantic similarity evaluation using Sentence Transformers and cosine similarity.

Supports lazy model loading, CPU/CUDA device resolution, batch embedding generation,
graceful error handling, and dependency injection of mock embedding providers for tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from src.utils.config import AppConfig, get_config, resolve_device


class EmbeddingBackendError(RuntimeError):
    """Raised when an embedding model fails to load or encode text."""


@dataclass(frozen=True)
class SemanticSimilarityResult:
    """Result of semantic similarity evaluation for one answer-reference pair."""

    score: Optional[float]
    status: str  # "computed", "skipped_empty", or "error"
    embedding_model: str
    error_message: Optional[str] = None


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute Cosine Similarity = (A · B) / (||A|| * ||B||) in [-1.0, 1.0].

    Returns 0.0 safely if either vector has zero L2 norm.
    Raises ValueError if vector dimensions do not match or vectors are empty.
    """
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()

    if a.size == 0 or b.size == 0:
        raise ValueError("Cannot compute cosine similarity for empty vectors.")
    if a.shape != b.shape:
        raise ValueError(f"Vector dimension mismatch: {a.shape} vs {b.shape}.")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    raw = float(np.dot(a, b) / (norm_a * norm_b))
    clipped = max(-1.0, min(1.0, raw))
    return round(clipped, 6)


def batch_cosine_similarity(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> List[float]:
    """Compute row-wise cosine similarities for two aligned 2D embedding matrices."""
    mat_a = np.asarray(embeddings_a, dtype=np.float64)
    mat_b = np.asarray(embeddings_b, dtype=np.float64)
    if mat_a.ndim != 2 or mat_b.ndim != 2:
        raise ValueError("Batch cosine similarity requires 2D embedding arrays.")
    if mat_a.shape != mat_b.shape:
        raise ValueError(f"Embedding matrix shape mismatch: {mat_a.shape} vs {mat_b.shape}.")

    return [cosine_similarity(mat_a[i], mat_b[i]) for i in range(mat_a.shape[0])]


class BaseEmbeddingProvider(ABC):
    """Abstract interface for embedding providers (real SentenceTransformers or mocks)."""

    model_name: str

    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode a sequence of strings into a 2D numpy float array of shape (N, D)."""


class SentenceTransformerEmbedder(BaseEmbeddingProvider):
    """Lazy-loading SentenceTransformer embedding provider with CPU fallback."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        preferred_device: str = "auto",
        app_config: Optional[AppConfig] = None,
        local_files_only: bool = False,
        model_loader: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.app_config = app_config or get_config()
        self.model_name = model_name or self.app_config.evaluation.embedding_model_id
        self.preferred_device = preferred_device
        self.local_files_only = local_files_only
        self._model_loader = model_loader
        self._model: Any = None
        self._device: Optional[str] = None

    def ensure_loaded(self) -> Any:
        """Lazily instantiate the SentenceTransformer model on first use."""
        if self._model is not None:
            return self._model

        try:
            self._device = resolve_device(self.preferred_device)
            if self._model_loader is not None:
                self._model = self._model_loader(
                    self.model_name,
                    device=self._device,
                    cache_folder=str(self.app_config.paths.hf_cache_dir / "sentence_transformers"),
                    local_files_only=self.local_files_only,
                )
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name,
                    device=self._device,
                    cache_folder=str(self.app_config.paths.hf_cache_dir / "sentence_transformers"),
                    local_files_only=self.local_files_only,
                )
        except Exception as exc:
            raise EmbeddingBackendError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

        return self._model

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings using lazy model loading."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self.ensure_loaded()
        try:
            vectors = model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            arr = np.asarray(vectors, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception as exc:
            raise EmbeddingBackendError(
                f"Failed to encode texts with '{self.model_name}': {exc}"
            ) from exc


class SemanticSimilarityEvaluator:
    """Computes cosine similarity between generated and reference answers."""

    def __init__(self, embedder: Optional[BaseEmbeddingProvider] = None) -> None:
        self.embedder: BaseEmbeddingProvider = embedder or SentenceTransformerEmbedder()

    def compute_similarity(self, generated: str, reference: str) -> SemanticSimilarityResult:
        """Compute semantic similarity for a single generated/reference pair."""
        return self.compute_batch_similarity([generated], [reference])[0]

    def compute_batch_similarity(
        self,
        generated_list: Sequence[str],
        reference_list: Sequence[str],
        batch_size: int = 32,
    ) -> List[SemanticSimilarityResult]:
        """
        Compute semantic similarity for aligned batches of generated and reference answers.

        Skips embedding calls for empty pairs and captures backend failures gracefully.
        """
        if len(generated_list) != len(reference_list):
            raise ValueError(
                f"Length mismatch between generated ({len(generated_list)}) "
                f"and reference ({len(reference_list)}) lists."
            )

        results: List[Optional[SemanticSimilarityResult]] = [None] * len(generated_list)
        valid_indices: List[int] = []
        valid_gen: List[str] = []
        valid_ref: List[str] = []

        for idx, (gen, ref) in enumerate(zip(generated_list, reference_list)):
            clean_gen = (gen or "").strip()
            clean_ref = (ref or "").strip()
            if not clean_gen or not clean_ref:
                results[idx] = SemanticSimilarityResult(
                    score=None,
                    status="skipped_empty",
                    embedding_model=self.embedder.model_name,
                    error_message="Cannot compute semantic similarity when generated or reference text is empty.",
                )
            else:
                valid_indices.append(idx)
                valid_gen.append(clean_gen)
                valid_ref.append(clean_ref)

        if valid_indices:
            try:
                gen_embs = self.embedder.encode(valid_gen, batch_size=batch_size)
                ref_embs = self.embedder.encode(valid_ref, batch_size=batch_size)
                scores = batch_cosine_similarity(gen_embs, ref_embs)
                for idx, score in zip(valid_indices, scores):
                    results[idx] = SemanticSimilarityResult(
                        score=score,
                        status="computed",
                        embedding_model=self.embedder.model_name,
                    )
            except Exception as exc:
                for idx in valid_indices:
                    results[idx] = SemanticSimilarityResult(
                        score=None,
                        status="error",
                        embedding_model=self.embedder.model_name,
                        error_message=str(exc),
                    )

        return [r for r in results if r is not None]
