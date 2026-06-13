"""情感分类的可选 BERT 微调对比实验。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .cli_utils import ChineseArgumentParser


LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def _load_optional_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "缺少 BERT 实验依赖。请先使用已安装 torch 和 transformers 的环境，"
            "或安装 requirements-bert.txt。"
        ) from exc

    return {
        "np": np,
        "pd": pd,
        "torch": torch,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "f1_score": f1_score,
        "train_test_split": train_test_split,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def sample_per_label(df: Any, per_label: int | None, seed: int) -> Any:
    if per_label is None or per_label <= 0:
        return df
    sampled_indexes: list[Any] = []
    for _label, indexes in df.groupby("label").groups.items():
        group = df.loc[indexes]
        sampled_indexes.extend(
            group.sample(
                n=min(len(group), per_label),
                random_state=seed,
            ).index.tolist()
        )
    return df.loc[sampled_indexes].reset_index(drop=True)


def load_training_frame(data_path: str | Path, per_label: int | None = None, seed: int = 42) -> Any:
    deps = _load_optional_dependencies()
    pd = deps["pd"]

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("训练数据必须包含 'text' 和 'label' 列")

    selected_columns = ["text", "label"]
    if "language" in df.columns:
        selected_columns.append("language")
    df = df[selected_columns].dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()
    invalid = sorted(set(df["label"]) - set(LABELS))
    if invalid:
        raise ValueError(f"不支持的标签：{invalid}。期望标签：{LABELS}")

    df = sample_per_label(df, per_label, seed)
    if len(df) < len(LABELS) * 2:
        raise ValueError("带标签样本不足，无法划分训练集和测试集")
    return df.reset_index(drop=True)


class ReviewDataset:
    def __init__(self, encodings: dict[str, Any], labels: list[int], torch_module: Any) -> None:
        self.encodings = encodings
        self.labels = labels
        self.torch = torch_module

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: self.torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = self.torch.tensor(self.labels[index])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def train_bert(
    data_path: str | Path = "data/processed/bilingual_reviews_train.csv",
    model_name: str = "bert-base-multilingual-cased",
    output_dir: str | Path = "outputs/bert_model_final",
    metrics_path: str | Path = "outputs/bert_metrics.json",
    epochs: float = 2.0,
    batch_size: int = 8,
    max_length: int = 160,
    learning_rate: float = 2e-5,
    seed: int = 42,
    per_label: int | None = None,
    validation_size: float = 0.15,
    test_size: float = 0.25,
) -> dict[str, Any]:
    """微调 BERT 并写入对比指标。"""

    deps = _load_optional_dependencies()
    np = deps["np"]
    torch = deps["torch"]
    train_test_split = deps["train_test_split"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    df = load_training_frame(data_path, per_label=per_label, seed=seed)
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1")
    if df["label"].value_counts().min() < 4:
        raise ValueError("Each label needs at least 4 samples for train/validation/test splits")

    stratify = df["label"] if df["label"].value_counts().min() > 1 else None
    train_pool_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    validation_fraction = validation_size / (1 - test_size)
    train_df, validation_df = train_test_split(
        train_pool_df,
        test_size=validation_fraction,
        random_state=seed,
        stratify=train_pool_df["label"],
    )

    tokenizer = deps["AutoTokenizer"].from_pretrained(model_name)
    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    test_encodings = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    validation_encodings = tokenizer(
        validation_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    train_labels = [LABEL_TO_ID[label] for label in train_df["label"].tolist()]
    validation_labels = [
        LABEL_TO_ID[label] for label in validation_df["label"].tolist()
    ]
    test_labels = [LABEL_TO_ID[label] for label in test_df["label"].tolist()]

    train_dataset = ReviewDataset(train_encodings, train_labels, torch)
    validation_dataset = ReviewDataset(
        validation_encodings,
        validation_labels,
        torch,
    )
    test_dataset = ReviewDataset(test_encodings, test_labels, torch)
    model = deps["AutoModelForSequenceClassification"].from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(deps["accuracy_score"](labels, predictions)),
            "macro_f1": float(
                deps["f1_score"](labels, predictions, average="macro", zero_division=0)
            ),
        }

    output_path = Path(output_dir)
    args = deps["TrainingArguments"](
        output_dir=str(output_path),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        seed=seed,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        save_total_limit=1,
        save_only_model=True,
        report_to="none",
    )
    trainer = deps["Trainer"](
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    predictions = trainer.predict(test_dataset)
    predicted_ids = np.argmax(predictions.predictions, axis=-1)
    y_true = [ID_TO_LABEL[int(label)] for label in test_labels]
    y_pred = [ID_TO_LABEL[int(label)] for label in predicted_ids]

    metrics = {
        "model_name": model_name,
        "training_device": str(args.device),
        "accuracy": float(deps["accuracy_score"](y_true, y_pred)),
        "macro_f1": float(deps["f1_score"](y_true, y_pred, average="macro", zero_division=0)),
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "split": {
            "validation_fraction": validation_size,
            "test_fraction": test_size,
            "seed": seed,
        },
        "data_path": str(data_path),
        "labels": LABELS,
        "classification_report": deps["classification_report"](
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
            output_dict=True,
        ),
    }
    if "language" in test_df.columns:
        language_metrics: dict[str, dict[str, float | int]] = {}
        for language in sorted(test_df["language"].astype(str).unique()):
            mask = test_df["language"].astype(str).eq(language).tolist()
            language_true = [label for label, selected in zip(y_true, mask) if selected]
            language_pred = [label for label, selected in zip(y_pred, mask) if selected]
            language_metrics[language] = {
                "accuracy": float(
                    deps["accuracy_score"](language_true, language_pred)
                ),
                "macro_f1": float(
                    deps["f1_score"](
                        language_true,
                        language_pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "support": len(language_true),
            }
        metrics["language_metrics"] = language_metrics

    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseArgumentParser(description="BERT 情感分类对比实验")
    parser.add_argument("--data", default="data/processed/bilingual_reviews_train.csv")
    parser.add_argument("--model-name", default="bert-base-multilingual-cased")
    parser.add_argument("--output-dir", default="outputs/bert_model_final")
    parser.add_argument("--metrics-path", default="outputs/bert_metrics.json")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-label", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.25)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metrics = train_bert(
        data_path=args.data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        metrics_path=args.metrics_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        seed=args.seed,
        per_label=args.sample_per_label,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    print(f"预训练模型={metrics['model_name']}")
    print(f"准确率={metrics['accuracy']:.4f}")
    print(f"宏平均F1={metrics['macro_f1']:.4f}")
    print(f"测试集数量={metrics['test_size']}")


if __name__ == "__main__":
    main()
