"""在带标签测试用例上运行规则、模型和混合流程的情感消融实验。"""

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
)
from src.preprocess import load_stopwords, preprocess_text


LABEL_ORDER = ["negative", "neutral", "positive"]
STATUS_COMPLETED = "完成"
STATUS_SKIPPED = "跳过"


def _safe_float(value: float) -> float:
    return float(value) if value == value else 0.0


def _evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
    skipped: int = 0,
) -> dict[str, Any]:
    if not y_true:
        return {
            "status": STATUS_SKIPPED,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "passed": 0,
            "failed": 0,
            "skipped": skipped,
            "classification_report": {},
        }

    passed = sum(1 for expected, actual in zip(y_true, y_pred) if expected == actual)
    failed = len(y_true) - passed
    return {
        "status": STATUS_COMPLETED,
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "macro_f1": _safe_float(
            f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        ),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            zero_division=0,
            output_dict=True,
        ),
    }


def _expected_cases(cases: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "expected_sentiment"}
    if not required.issubset(set(cases.columns)):
        raise ValueError("消融实验用例必须包含 'text' 和 'expected_sentiment' 列")

    valid = cases.copy()
    valid["text"] = valid["text"].fillna("").astype(str)
    valid["expected_sentiment"] = valid["expected_sentiment"].fillna("").astype(str).str.strip()
    valid = valid[valid["text"].str.strip() != ""]
    valid = valid[valid["expected_sentiment"].isin(LABEL_ORDER)]
    return valid.reset_index(drop=True)


def _model_only_predictions(
    texts: list[str],
    model_dir: str | Path,
) -> tuple[list[str], bool]:
    model, vectorizer = load_model_if_available(model_dir)
    if model is None or vectorizer is None:
        return [], False

    stopwords = load_stopwords()
    processed = [preprocess_text(text, stopwords=stopwords) for text in texts]
    features = vectorizer.transform(processed)
    return [str(label) for label in model.predict(features)], True


def _error_rows(
    experiment: str,
    cases: pd.DataFrame,
    predictions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in cases.iterrows():
        expected = str(row["expected_sentiment"])
        actual = predictions[index] if index < len(predictions) else ""
        if expected == actual:
            continue
        rows.append(
            {
                "实验版本": experiment,
                "编号": row.get("id", ""),
                "评价文本": row["text"],
                "预期情感": expected,
                "实际情感": actual,
            }
        )
    return rows


def run_ablation(
    cases: pd.DataFrame,
    output_dir: str | Path = "outputs/reports",
    model_dir: str | Path = "models",
) -> dict[str, Any]:
    """运行消融实验并写入指标和错误样例文件。"""

    valid_cases = _expected_cases(cases)
    texts = valid_cases["text"].tolist()
    expected = valid_cases["expected_sentiment"].tolist()

    rule_predictions = [rule_based_sentiment(text)[0] for text in texts]
    hybrid_predictions = [
        result["sentiment"]
        for result in analyze_batch(texts, use_llm=False)
    ]
    model_predictions, model_available = _model_only_predictions(texts, model_dir)
    bert_results = (
        bert_based_sentiments(texts)
        if configured_sentiment_backend() in {"auto", "bert"}
        else None
    )
    bert_predictions = (
        [str(result[0]) for result in bert_results]
        if bert_results is not None
        else []
    )

    experiments = {
        "rule-only": _evaluate_predictions(expected, rule_predictions),
        "model-only": (
            _evaluate_predictions(expected, model_predictions)
            if model_available
            else {
                "status": STATUS_SKIPPED,
                "reason": f"在 {model_dir} 中未找到模型文件",
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "passed": 0,
                "failed": 0,
                "skipped": len(expected),
                "classification_report": {},
            }
        ),
        "bert-only": (
            _evaluate_predictions(expected, bert_predictions)
            if bert_results is not None
            else {
                "status": STATUS_SKIPPED,
                "reason": "BERT model or dependencies are unavailable",
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "passed": 0,
                "failed": 0,
                "skipped": len(expected),
                "classification_report": {},
            }
        ),
        "hybrid": _evaluate_predictions(expected, hybrid_predictions),
    }

    errors = []
    errors.extend(_error_rows("rule-only", valid_cases, rule_predictions))
    if model_available:
        errors.extend(_error_rows("model-only", valid_cases, model_predictions))
    if bert_results is not None:
        errors.extend(_error_rows("bert-only", valid_cases, bert_predictions))
    errors.extend(_error_rows("hybrid", valid_cases, hybrid_predictions))

    metrics = {
        "data_size": len(valid_cases),
        "model_dir": str(model_dir),
        "experiments": experiments,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "ablation_metrics.json").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as metrics_file:
        metrics_file.write(json.dumps(metrics, ensure_ascii=False, indent=2))
    pd.DataFrame(
        errors,
        columns=["实验版本", "编号", "评价文本", "预期情感", "实际情感"],
    ).to_csv(
        output_path / "ablation_errors.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    return metrics


def main() -> None:
    parser = ChineseArgumentParser(description="运行情感分类消融实验")
    parser.add_argument("--cases", default="data/test_cases.csv")
    parser.add_argument("--output-dir", default="outputs/reports")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    metrics = run_ablation(
        pd.read_csv(args.cases),
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
    for name, values in metrics["experiments"].items():
        if values["status"] == STATUS_SKIPPED:
            print(f"{name}: 跳过（{values.get('reason', '')}）")
            continue
        print(
            f"{name}: 准确率={values['accuracy']:.4f}，"
            f"宏平均F1={values['macro_f1']:.4f}，"
            f"通过={values['passed']}，失败={values['failed']}"
        )


if __name__ == "__main__":
    main()
