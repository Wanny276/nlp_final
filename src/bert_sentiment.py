"""Lazy, optional BERT inference for sentiment classification."""

from __future__ import annotations

import os
import threading
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None


LABELS = {"negative", "neutral", "positive"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_project_env() -> None:
    if load_dotenv is None:
        return
    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.exists():
        dotenv_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=dotenv_path)


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _resolve_model_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class BertConfig:
    model_path: Path
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 160

    @classmethod
    def from_env(cls) -> "BertConfig":
        _load_project_env()
        return cls(
            model_path=_resolve_model_path(
                os.getenv("BERT_MODEL_PATH", "outputs/bert_model")
            ),
            device=os.getenv("BERT_DEVICE", "auto").strip().lower() or "auto",
            batch_size=_positive_int("BERT_BATCH_SIZE", 32),
            max_length=_positive_int("BERT_MAX_LENGTH", 160),
        )


@dataclass(frozen=True)
class BertPrediction:
    label: str
    confidence: float
    device: str


class BertSentimentPredictor:
    """Load a local fine-tuned model once and run batched inference."""

    def __init__(self, config: BertConfig) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (config.model_path / "config.json").exists():
            raise FileNotFoundError(f"BERT model not found: {config.model_path}")

        self.torch = torch
        self.config = config
        self.device = self._select_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.model_path),
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(config.model_path),
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.id_to_label = self._label_mapping(self.model.config.id2label)
        self._inference_lock = threading.Lock()

    def _select_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("BERT_DEVICE=cuda but CUDA is not available")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("BERT_DEVICE must be auto, cpu, or cuda")
        return requested

    @staticmethod
    def _label_mapping(mapping: dict[Any, Any]) -> dict[int, str]:
        normalized = {int(key): str(value).lower() for key, value in mapping.items()}
        if set(normalized.values()) != LABELS:
            raise ValueError(f"Unsupported BERT labels: {normalized}")
        return normalized

    def predict(self, texts: list[str]) -> list[BertPrediction]:
        if not texts:
            return []

        predictions: list[BertPrediction] = []
        with self._inference_lock:
            for start in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[start : start + self.config.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with self.torch.inference_mode():
                    probabilities = self.torch.softmax(
                        self.model(**encoded).logits,
                        dim=-1,
                    )
                confidence, predicted_ids = probabilities.max(dim=-1)
                predictions.extend(
                    BertPrediction(
                        label=self.id_to_label[int(label_id)],
                        confidence=float(score),
                        device=self.device,
                    )
                    for label_id, score in zip(
                        predicted_ids.detach().cpu().tolist(),
                        confidence.detach().cpu().tolist(),
                    )
                )
        return predictions


_PREDICTOR: BertSentimentPredictor | None = None
_PREDICTOR_KEY: tuple[str, str, int, int, int, int] | None = None
_PREDICTOR_ERROR: str | None = None
_PREDICTOR_LOCK = threading.Lock()


def _config_key(config: BertConfig) -> tuple[str, str, int, int, int, int]:
    config_path = config.model_path / "config.json"
    weight_candidates = [
        config.model_path / "model.safetensors",
        config.model_path / "pytorch_model.bin",
    ]
    config_version = config_path.stat().st_mtime_ns if config_path.exists() else 0
    weight_version = max(
        (path.stat().st_mtime_ns for path in weight_candidates if path.exists()),
        default=0,
    )
    return (
        str(config.model_path.resolve()),
        config.device,
        config.batch_size,
        config.max_length,
        config_version,
        weight_version,
    )


def get_predictor(config: BertConfig | None = None) -> BertSentimentPredictor | None:
    """Return the shared predictor, or None when BERT is unavailable."""

    global _PREDICTOR, _PREDICTOR_KEY, _PREDICTOR_ERROR

    config = config or BertConfig.from_env()
    key = _config_key(config)
    if _PREDICTOR_KEY == key:
        return _PREDICTOR

    with _PREDICTOR_LOCK:
        if _PREDICTOR_KEY == key:
            return _PREDICTOR
        try:
            predictor = BertSentimentPredictor(config)
        except Exception as exc:
            _PREDICTOR = None
            _PREDICTOR_ERROR = f"{exc.__class__.__name__}: {exc}"
        else:
            _PREDICTOR = predictor
            _PREDICTOR_ERROR = None
        _PREDICTOR_KEY = key
        return _PREDICTOR


def predict_bert(
    texts: list[str],
    config: BertConfig | None = None,
) -> list[BertPrediction] | None:
    predictor = get_predictor(config)
    if predictor is None:
        return None
    try:
        return predictor.predict(texts)
    except Exception as exc:
        global _PREDICTOR_ERROR
        _PREDICTOR_ERROR = f"{exc.__class__.__name__}: {exc}"
        if predictor.device == "cuda":
            try:
                predictor.torch.cuda.empty_cache()
            except Exception:
                pass
        return None


def bert_status(config: BertConfig | None = None) -> dict[str, Any]:
    config = config or BertConfig.from_env()
    dependencies_available = (
        find_spec("torch") is not None
        and find_spec("transformers") is not None
    )
    return {
        "model_path": str(config.model_path),
        "model_available": (config.model_path / "config.json").exists(),
        "dependencies_available": dependencies_available,
        "ready": (
            (config.model_path / "config.json").exists()
            and dependencies_available
            and _PREDICTOR_ERROR is None
        ),
        "loaded": _PREDICTOR is not None and _PREDICTOR_KEY == _config_key(config),
        "device": _PREDICTOR.device if _PREDICTOR is not None else config.device,
        "error": _PREDICTOR_ERROR,
    }


def reset_bert_cache() -> None:
    """Clear process-local state. Intended for tests and configuration reloads."""

    global _PREDICTOR, _PREDICTOR_KEY, _PREDICTOR_ERROR
    _PREDICTOR = None
    _PREDICTOR_KEY = None
    _PREDICTOR_ERROR = None
