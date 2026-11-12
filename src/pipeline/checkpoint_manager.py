"""
Atomic checkpointing, corruption detection, and compatibility-validated resume support.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.evaluation.evaluator import EvaluationRecord
from src.generation.generator import GenerationResult
from src.hallucination.claim_verifier import ClaimVerificationResult
from src.hallucination.faithfulness import FaithfulnessMetrics
from src.hallucination.hallucination_detector import HallucinationDetectionRecord
from src.judge.judge_evaluator import JudgeEvaluationRecord


class CheckpointError(RuntimeError):
    """Base error for checkpoint handling failures."""


class CheckpointCorruptedError(CheckpointError):
    """Raised when an existing checkpoint file is malformed or unreadable."""


class CheckpointMismatchError(CheckpointError):
    """Raised when resuming a run with incompatible models, dataset, or prompt settings."""


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via temporary file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _read_json_list(path: Path, label: str) -> List[Dict[str, Any]]:
    """Read and validate a JSON list checkpoint file."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        raise CheckpointCorruptedError(
            f"Corrupted {label} checkpoint at '{path}': {exc}"
        ) from exc

    if not isinstance(data, list):
        raise CheckpointCorruptedError(
            f"Invalid {label} checkpoint format at '{path}': expected JSON list."
        )
    return data


class CheckpointManager:
    """Manages incremental stage checkpoints and resume validation inside a run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_config_path = self.run_dir / "run_config.json"
        self.generation_path = self.run_dir / "generation_results.json"
        self.evaluation_path = self.run_dir / "evaluation_results.json"
        self.hallucination_path = self.run_dir / "hallucination_results.json"
        self.judge_path = self.run_dir / "judge_results.json"

    def initialize_or_validate(
        self,
        serialized_config: Dict[str, Any],
        fingerprint: Dict[str, Any],
        resume: bool,
    ) -> None:
        """
        Save run_config.json on a fresh run, or verify compatibility when resuming.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if not self.run_config_path.exists():
            _atomic_write_json(
                self.run_config_path,
                {
                    "config": serialized_config,
                    "fingerprint": fingerprint,
                },
            )
            return

        if not resume:
            # Fresh run with explicit reused run_id only allowed if directory has no prior config,
            # otherwise raise to prevent accidental overwriting of a completed run.
            raise CheckpointMismatchError(
                f"Run directory '{self.run_dir}' already exists with run_config.json. "
                "Set resume=True to resume or provide a new run_id."
            )

        try:
            with self.run_config_path.open("r", encoding="utf-8") as fh:
                saved = json.load(fh)
            saved_fp = saved["fingerprint"]
        except Exception as exc:
            raise CheckpointCorruptedError(
                f"Corrupted run_config.json at '{self.run_config_path}': {exc}"
            ) from exc

        critical_keys = (
            "dataset_hash",
            "max_questions",
            "question_ids",
            "models",
            "prompt_template_name",
            "system_prompt",
            "generation_config",
            "embedding_model_id",
            "nli_model_id",
        )
        mismatches: List[str] = []
        for key in critical_keys:
            if saved_fp.get(key) != fingerprint.get(key):
                mismatches.append(key)

        if fingerprint.get("enable_judge") and saved_fp.get("enable_judge"):
            for jkey in ("judge_model", "judge_prompt_version"):
                if saved_fp.get(jkey) != fingerprint.get(jkey):
                    mismatches.append(jkey)

        if mismatches:
            raise CheckpointMismatchError(
                "Cannot resume run because configuration differs on: "
                + ", ".join(mismatches)
            )

        # Update stored config if new optional evaluation stages were turned on cleanly
        _atomic_write_json(
            self.run_config_path,
            {
                "config": serialized_config,
                "fingerprint": fingerprint,
            },
        )

    def load_generation_checkpoint(self) -> Dict[Tuple[str, str], GenerationResult]:
        """Load generation results keyed by (model_id, question_id)."""
        raw = _read_json_list(self.generation_path, "generation_results")
        loaded: Dict[Tuple[str, str], GenerationResult] = {}
        try:
            for item in raw:
                rec = GenerationResult(**item)
                loaded[(rec.model_id, rec.question_id)] = rec
        except Exception as exc:
            raise CheckpointCorruptedError(
                f"Corrupted GenerationResult entry in '{self.generation_path}': {exc}"
            ) from exc
        return loaded

    def save_generation_checkpoint(self, records_map: Mapping[Tuple[str, str], GenerationResult]) -> None:
        """Atomically persist generation results in deterministic order."""
        ordered = [records_map[k].to_dict() for k in sorted(records_map.keys())]
        _atomic_write_json(self.generation_path, ordered)

    def load_evaluation_checkpoint(self) -> Dict[Tuple[str, str], EvaluationRecord]:
        """Load Phase 5 evaluation records keyed by (model_id, question_id)."""
        raw = _read_json_list(self.evaluation_path, "evaluation_results")
        loaded: Dict[Tuple[str, str], EvaluationRecord] = {}
        try:
            for item in raw:
                rec = EvaluationRecord(**item)
                loaded[(rec.model_id, rec.question_id)] = rec
        except Exception as exc:
            raise CheckpointCorruptedError(
                f"Corrupted EvaluationRecord entry in '{self.evaluation_path}': {exc}"
            ) from exc
        return loaded

    def save_evaluation_checkpoint(self, records_map: Mapping[Tuple[str, str], EvaluationRecord]) -> None:
        """Atomically persist Phase 5 evaluation records."""
        ordered = [records_map[k].to_dict() for k in sorted(records_map.keys())]
        _atomic_write_json(self.evaluation_path, ordered)

    def load_hallucination_checkpoint(self) -> Dict[Tuple[str, str], HallucinationDetectionRecord]:
        """Load Phase 6 hallucination detection records keyed by (model_id, question_id)."""
        raw = _read_json_list(self.hallucination_path, "hallucination_results")
        loaded: Dict[Tuple[str, str], HallucinationDetectionRecord] = {}
        try:
            for entry in raw:
                claims = [ClaimVerificationResult(**c) for c in entry.get("claims", [])]
                faith = FaithfulnessMetrics(**entry["faithfulness"])
                rec = HallucinationDetectionRecord(
                    question_id=entry["question_id"],
                    model_id=entry["model_id"],
                    category=entry.get("category", ""),
                    hallucination_test_type=entry.get("hallucination_test_type", ""),
                    expected_behavior=entry.get("expected_behavior", ""),
                    has_context=entry["has_context"],
                    is_counterfactual_context=entry.get("is_counterfactual_context", False),
                    generated_answer=entry.get("generated_answer", ""),
                    context=entry.get("context", ""),
                    reference_answer=entry.get("reference_answer", ""),
                    claims=claims,
                    faithfulness=faith,
                    hallucination_detected=entry["hallucination_detected"],
                    hallucination_types=entry.get("hallucination_types", []),
                    context_hallucination_rate=entry.get("context_hallucination_rate"),
                    detection_status=entry["detection_status"],
                    explanation=entry.get("explanation", ""),
                    metadata=entry.get("metadata", {}),
                )
                loaded[(rec.model_id, rec.question_id)] = rec
        except Exception as exc:
            raise CheckpointCorruptedError(
                f"Corrupted HallucinationDetectionRecord entry in '{self.hallucination_path}': {exc}"
            ) from exc
        return loaded

    def save_hallucination_checkpoint(
        self, records_map: Mapping[Tuple[str, str], HallucinationDetectionRecord]
    ) -> None:
        """Atomically persist Phase 6 hallucination records."""
        ordered = [records_map[k].to_dict() for k in sorted(records_map.keys())]
        _atomic_write_json(self.hallucination_path, ordered)

    def load_judge_checkpoint(self) -> Dict[Tuple[str, str], JudgeEvaluationRecord]:
        """Load Phase 7 judge records keyed by (evaluated_model_id, question_id)."""
        raw = _read_json_list(self.judge_path, "judge_results")
        loaded: Dict[Tuple[str, str], JudgeEvaluationRecord] = {}
        try:
            for item in raw:
                rec = JudgeEvaluationRecord(**item)
                loaded[(rec.evaluated_model_id, rec.question_id)] = rec
        except Exception as exc:
            raise CheckpointCorruptedError(
                f"Corrupted JudgeEvaluationRecord entry in '{self.judge_path}': {exc}"
            ) from exc
        return loaded

    def save_judge_checkpoint(
        self, records_map: Mapping[Tuple[str, str], JudgeEvaluationRecord]
    ) -> None:
        """Atomically persist Phase 7 judge records."""
        ordered = [records_map[k].to_dict() for k in sorted(records_map.keys())]
        _atomic_write_json(self.judge_path, ordered)
