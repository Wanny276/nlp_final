"""Evaluate sentiment backends on the held-out robustness stress suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.cli_utils import ChineseArgumentParser
from src.nlp_analyzer import (
    analyze_batch,
    bert_based_sentiments,
    configured_sentiment_backend,
    load_model_if_available,
    rule_based_sentiment,
    sentiment_runtime_status,
)
from src.preprocess import load_stopwords, preprocess_text


LABEL_ORDER = ["negative", "neutral", "positive"]
REQUIRED_COLUMNS = {
    "id",
    "category",
    "language",
    "text",
    "expected_sentiment",
}


def load_stress_cases(path: str | Path) -> pd.DataFrame:
    cases = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(cases.columns)
    if missing:
        raise ValueError(f"压力测试集缺少列：{sorted(missing)}")

    cases = cases.copy()
    for column in REQUIRED_COLUMNS:
        cases[column] = cases[column].fillna("").astype(str).str.strip()
    cases = cases[cases["text"] != ""].reset_index(drop=True)

    invalid_labels = sorted(set(cases["expected_sentiment"]) - set(LABEL_ORDER))
    if invalid_labels:
        raise ValueError(f"压力测试集包含无效标签：{invalid_labels}")
    if cases["id"].duplicated().any():
        raise ValueError("压力测试集包含重复 id")
    if cases["text"].str.casefold().duplicated().any():
        raise ValueError("压力测试集包含重复文本")
    return cases


def _safe_float(value: float) -> float:
    return float(value) if value == value else 0.0


def evaluate_predictions(
    expected: list[str],
    predicted: list[str],
) -> dict[str, Any]:
    passed = sum(
        actual == target
        for target, actual in zip(expected, predicted)
    )
    return {
        "accuracy": _safe_float(accuracy_score(expected, predicted)),
        "macro_f1": _safe_float(
            f1_score(
                expected,
                predicted,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "passed": passed,
        "failed": len(expected) - passed,
        "classification_report": classification_report(
            expected,
            predicted,
            labels=LABEL_ORDER,
            zero_division=0,
            output_dict=True,
        ),
    }


def grouped_metrics(
    cases: pd.DataFrame,
    predictions: list[str],
    column: str,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for value in sorted(cases[column].unique()):
        indexes = cases.index[cases[column] == value].tolist()
        expected = [
            str(cases.loc[index, "expected_sentiment"])
            for index in indexes
        ]
        predicted = [predictions[index] for index in indexes]
        group = evaluate_predictions(expected, predicted)
        metrics[str(value)] = {
            "support": len(indexes),
            "accuracy": group["accuracy"],
            "macro_f1": group["macro_f1"],
            "passed": group["passed"],
            "failed": group["failed"],
        }
    return metrics


def _tfidf_predictions(
    texts: list[str],
    model_dir: str | Path,
) -> list[str] | None:
    model, vectorizer = load_model_if_available(model_dir)
    if model is None or vectorizer is None:
        return None
    stopwords = load_stopwords()
    processed = [
        preprocess_text(text, stopwords=stopwords)
        for text in texts
    ]
    return [str(label) for label in model.predict(vectorizer.transform(processed))]


def _experiment_metrics(
    cases: pd.DataFrame,
    predictions: list[str] | None,
    reason: str = "",
) -> dict[str, Any]:
    if predictions is None:
        return {
            "status": "跳过",
            "reason": reason,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "passed": 0,
            "failed": 0,
            "by_category": {},
            "by_language": {},
        }

    expected = cases["expected_sentiment"].tolist()
    metrics = evaluate_predictions(expected, predictions)
    metrics["status"] = "完成"
    metrics["by_category"] = grouped_metrics(cases, predictions, "category")
    metrics["by_language"] = grouped_metrics(cases, predictions, "language")
    return metrics


def run_stress_test(
    cases: pd.DataFrame,
    output_dir: str | Path = "outputs/reports/stress_test",
    model_dir: str | Path = "models",
) -> dict[str, Any]:
    texts = cases["text"].tolist()
    expected = cases["expected_sentiment"].tolist()

    rule_predictions = [rule_based_sentiment(text)[0] for text in texts]
    tfidf_predictions = _tfidf_predictions(texts, model_dir)
    bert_results = (
        bert_based_sentiments(texts)
        if configured_sentiment_backend() in {"auto", "bert"}
        else None
    )
    bert_predictions = (
        [str(result[0]) for result in bert_results]
        if bert_results is not None
        else None
    )
    hybrid_results = analyze_batch(texts, use_llm=False)
    hybrid_predictions = [
        str(result["sentiment"])
        for result in hybrid_results
    ]

    predictions = cases[
        ["id", "category", "language", "text", "expected_sentiment"]
    ].copy()
    predictions["rule_only"] = rule_predictions
    predictions["tfidf_only"] = (
        tfidf_predictions if tfidf_predictions is not None else ""
    )
    predictions["bert_only"] = (
        bert_predictions if bert_predictions is not None else ""
    )
    predictions["hybrid"] = hybrid_predictions
    predictions["hybrid_confidence"] = [
        result["confidence"]
        for result in hybrid_results
    ]
    predictions["hybrid_source"] = [
        result["sentiment_source"]
        for result in hybrid_results
    ]
    predictions["bert_chunk_count"] = [
        result.get("sentiment_chunk_count", 0)
        for result in hybrid_results
    ]
    predictions["long_text_truncated"] = [
        result.get("long_text_truncated", False)
        for result in hybrid_results
    ]

    experiment_predictions = {
        "rule-only": rule_predictions,
        "tfidf-only": tfidf_predictions,
        "bert-only": bert_predictions,
        "hybrid": hybrid_predictions,
    }
    experiments = {
        name: _experiment_metrics(
            cases,
            values,
            reason=(
                f"在 {model_dir} 中未找到传统模型"
                if name == "tfidf-only"
                else "BERT 模型或依赖不可用"
            ),
        )
        for name, values in experiment_predictions.items()
    }

    error_rows: list[dict[str, Any]] = []
    for experiment, values in experiment_predictions.items():
        if values is None:
            continue
        for index, (target, actual) in enumerate(zip(expected, values)):
            if target == actual:
                continue
            row = cases.iloc[index]
            error_rows.append(
                {
                    "experiment": experiment,
                    "id": row["id"],
                    "category": row["category"],
                    "language": row["language"],
                    "text": row["text"],
                    "expected_sentiment": target,
                    "actual_sentiment": actual,
                }
            )

    runtime = sentiment_runtime_status()
    metrics = {
        "data_size": len(cases),
        "categories": cases["category"].value_counts().sort_index().to_dict(),
        "label_distribution": (
            cases["expected_sentiment"].value_counts().sort_index().to_dict()
        ),
        "language_distribution": (
            cases["language"].value_counts().sort_index().to_dict()
        ),
        "bert_model_path": runtime["bert"]["model_path"],
        "experiments": experiments,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "stress_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions.to_csv(
        output_path / "stress_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    pd.DataFrame(error_rows).to_csv(
        output_path / "stress_errors.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    return metrics


def main() -> None:
    parser = ChineseArgumentParser(description="运行独立情感压力测试")
    parser.add_argument("--cases", default="data/stress_test_cases.csv")
    parser.add_argument(
        "--output-dir",
        default="outputs/reports/stress_test_final",
    )
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    cases = load_stress_cases(args.cases)
    metrics = run_stress_test(
        cases,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
    print(
        f"测试集={metrics['data_size']} 条，"
        f"类别={len(metrics['categories'])} 类"
    )
    for name, values in metrics["experiments"].items():
        if values["status"] == "跳过":
            print(f"{name}: 跳过（{values['reason']}）")
            continue
        print(
            f"{name}: 准确率={values['accuracy']:.4f}，"
            f"宏平均F1={values['macro_f1']:.4f}，"
            f"通过={values['passed']}，失败={values['failed']}"
        )


if __name__ == "__main__":
    main()
