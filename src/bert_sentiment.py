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
    stride: int = 32
    max_chunks: int = 16

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
            stride=_positive_int("BERT_STRIDE", 32),
            max_chunks=_positive_int("BERT_MAX_CHUNKS", 16),
        )


@dataclass(frozen=True)
class BertPrediction:
    label: str
    confidence: float
    device: str
    chunk_count: int = 1
    token_count: int = 0
    truncated: bool = False


def _split_token_ids(
    token_ids: list[int],
    content_limit: int,
    stride: int,
    max_chunks: int,
) -> tuple[list[list[int]], bool]:
    """Split token ids into overlapping windows and report capped input."""

    if content_limit <= 0:
        raise ValueError("BERT_MAX_LENGTH is too small for tokenizer special tokens")
    if max_chunks <= 0:
        raise ValueError("BERT_MAX_CHUNKS must be positive")
    if not token_ids:
        return [[]], False

    overlap = min(max(stride, 0), content_limit - 1)
    step = content_limit - overlap
    chunks: list[list[int]] = []
    start = 0
    while start < len(token_ids):
        chunk = token_ids[start : start + content_limit]
        chunks.append(chunk)
        if start + len(chunk) >= len(token_ids):
            return chunks, False
        if len(chunks) >= max_chunks:
            return chunks, True
        start += step
    return chunks, False


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

    def _encode_chunks(
        self,
        text: str,
    ) -> tuple[list[dict[str, list[int]]], int, bool]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
        token_ids = list(encoded["input_ids"])
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        content_limit = self.config.max_length - special_tokens
        chunks, truncated = _split_token_ids(
            token_ids,
            content_limit=content_limit,
            stride=self.config.stride,
            max_chunks=self.config.max_chunks,
        )

        features: list[dict[str, list[int]]] = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        if cls_token_id is None or sep_token_id is None:
            raise ValueError("BERT tokenizer must define CLS and SEP token ids")
        for chunk in chunks:
            input_ids = [cls_token_id, *chunk, sep_token_id]
            feature = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
            if "token_type_ids" in self.tokenizer.model_input_names:
                feature["token_type_ids"] = [0] * len(input_ids)
            features.append(feature)
        return features, len(token_ids), truncated

    def predict(self, texts: list[str]) -> list[BertPrediction]:
        if not texts:
            return []

        predictions: list[BertPrediction] = []
        with self._inference_lock:
            for start in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[start : start + self.config.batch_size]
                chunk_features: list[dict[str, list[int]]] = []
                chunk_owners: list[int] = []
                chunk_weights: list[int] = []
                sample_metadata: list[tuple[int, int, bool]] = []

                for sample_index, text in enumerate(batch_texts):
                    features, token_count, truncated = self._encode_chunks(text)
                    sample_metadata.append(
                        (len(features), token_count, truncated)
                    )
                    for feature in features:
                        chunk_features.append(feature)
                        chunk_owners.append(sample_index)
                        special_tokens = (
                            self.tokenizer.num_special_tokens_to_add(pair=False)
                        )
                        chunk_weights.append(
                            max(1, len(feature["input_ids"]) - special_tokens)
                        )

                probability_sums: list[list[float] | None] = [
                    None for _ in batch_texts
                ]
                weight_sums = [0 for _ in batch_texts]
                for chunk_start in range(
                    0,
                    len(chunk_features),
                    self.config.batch_size,
                ):
                    chunk_end = chunk_start + self.config.batch_size
                    encoded_chunks = self.tokenizer.pad(
                        chunk_features[chunk_start:chunk_end],
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)
                    with self.torch.inference_mode():
                        chunk_probabilities = self.torch.softmax(
                            self.model(**encoded_chunks).logits,
                            dim=-1,
                        )

                    for probabilities, owner, weight in zip(
                        chunk_probabilities.detach().cpu().tolist(),
                        chunk_owners[chunk_start:chunk_end],
                        chunk_weights[chunk_start:chunk_end],
                    ):
                        if probability_sums[owner] is None:
                            probability_sums[owner] = [
                                0.0 for _ in probabilities
                            ]
                        for index, probability in enumerate(probabilities):
                            probability_sums[owner][index] += probability * weight
                        weight_sums[owner] += weight

                for sample_index, metadata in enumerate(sample_metadata):
                    chunk_count, token_count, truncated = metadata
                    weighted = probability_sums[sample_index]
                    if weighted is None or weight_sums[sample_index] <= 0:
                        raise RuntimeError("BERT produced no chunk probabilities")
                    averaged = [
                        probability / weight_sums[sample_index]
                        for probability in weighted
                    ]
                    predicted_id = max(
                        range(len(averaged)),
                        key=averaged.__getitem__,
                    )
                    predictions.append(
                        BertPrediction(
                            label=self.id_to_label[predicted_id],
                            confidence=float(averaged[predicted_id]),
                            device=self.device,
                            chunk_count=chunk_count,
                            token_count=token_count,
                            truncated=truncated,
                        )
                    )
        return predictions


_PREDICTOR: BertSentimentPredictor | None = None
_PREDICTOR_KEY: tuple[str, str, int, int, int, int, int, int] | None = None
_PREDICTOR_ERROR: str | None = None
_PREDICTOR_LOCK = threading.Lock()


def _config_key(
    config: BertConfig,
) -> tuple[str, str, int, int, int, int, int, int]:
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
        config.stride,
        config.max_chunks,
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
        "max_length": config.max_length,
        "stride": config.stride,
        "max_chunks": config.max_chunks,
        "error": _PREDICTOR_ERROR,
    }


def reset_bert_cache() -> None:
    """Clear process-local state. Intended for tests and configuration reloads."""

    global _PREDICTOR, _PREDICTOR_KEY, _PREDICTOR_ERROR
    _PREDICTOR = None
    _PREDICTOR_KEY = None
    _PREDICTOR_ERROR = None
