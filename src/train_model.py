"""Train and compare TF-IDF based sentiment classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .preprocess import load_stopwords, preprocess_many


LABEL_ORDER = ["positive", "neutral", "negative"]


def _metric_summary(y_true: list[str], y_pred: list[str]) -> dict[str, float | int | None]:
    from sklearn.metrics import accuracy_score, f1_score

    if not y_true:
        return {"accuracy": None, "macro_f1": None, "test_size": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_size": len(y_true),
    }


def _plot_confusion_matrix(matrix: list[list[int]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABEL_ORDER)), LABEL_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_model_comparison(results: dict[str, dict[str, float]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results)
    positions = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(
        positions - width / 2,
        [results[name]["accuracy"] for name in names],
        width,
        label="Accuracy",
    )
    ax.bar(
        positions + width / 2,
        [results[name]["macro_f1"] for name in names],
        width,
        label="Macro-F1",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(positions, names, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train(
    data_path: str | Path,
    model_dir: str | Path = "models",
    report_dir: str | Path = "outputs/reports",
    chart_dir: str | Path = "outputs/charts",
) -> dict[str, Any]:
    """Train several classifiers and save the best one."""

    import joblib
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Training data must contain 'text' and 'label' columns")

    stopwords = load_stopwords()
    texts = preprocess_many(df["text"].fillna("").tolist(), stopwords=stopwords)
    labels = df["label"].fillna("neutral").tolist()
    languages = (
        df["language"].fillna("unknown").astype(str).tolist()
        if "language" in df.columns
        else None
    )

    stratify = labels if len(set(labels)) > 1 and min(labels.count(label) for label in set(labels)) > 1 else None
    if languages is None:
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
        lang_test = None
    else:
        x_train, x_test, y_train, y_test, _, lang_test = train_test_split(
            texts,
            labels,
            languages,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    candidates = {
        "Dummy Most Frequent": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
    }

    results: dict[str, dict[str, float]] = {}
    trained_models = {}
    for name, model in candidates.items():
        model.fit(x_train_vec, y_train)
        predictions = model.predict(x_test_vec)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        }
        trained_models[name] = model

    best_name = max(results, key=lambda name: (results[name]["macro_f1"], results[name]["accuracy"]))
    best_metrics = results[best_name]
    best_predictions = trained_models[best_name].predict(x_test_vec).tolist()
    report = classification_report(
        y_test,
        best_predictions,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, best_predictions, labels=LABEL_ORDER).tolist()

    language_metrics = {
        "overall": _metric_summary(list(y_test), best_predictions),
    }
    if lang_test is not None:
        for language in sorted(set(languages or [])):
            indices = [index for index, value in enumerate(lang_test) if value == language]
            language_metrics[language] = _metric_summary(
                [y_test[index] for index in indices],
                [best_predictions[index] for index in indices],
            )

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models[best_name], output_dir / "sentiment_model.pkl")
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")

    report_output_dir = Path(report_dir)
    chart_output_dir = Path(chart_dir)
    report_output_dir.mkdir(parents=True, exist_ok=True)
    chart_output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "best_model": best_name,
        "accuracy": best_metrics["accuracy"],
        "macro_f1": best_metrics["macro_f1"],
        "results": results,
        "classification_report": report,
        "confusion_matrix": matrix,
        "labels": LABEL_ORDER,
        "language_metrics": language_metrics,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "data_path": str(data_path),
    }
    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_output_dir / "classification_report.json").write_text(
        json.dumps(
            {
                "best_model": best_name,
                "labels": LABEL_ORDER,
                "classification_report": report,
                "confusion_matrix": matrix,
                "language_metrics": language_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_confusion_matrix(matrix, chart_output_dir / "confusion_matrix.png")
    _plot_model_comparison(results, chart_output_dir / "model_comparison.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_reviews.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report-dir", default="outputs/reports")
    parser.add_argument("--chart-dir", default="outputs/charts")
    args = parser.parse_args()

    metrics = train(args.data, args.model_dir, args.report_dir, args.chart_dir)
    print("model comparison:")
    for name, result in metrics["results"].items():
        print(f"- {name}: accuracy={result['accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}")
    print(f"best_model={metrics['best_model']}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print("language metrics:")
    for language, values in metrics["language_metrics"].items():
        accuracy = values["accuracy"]
        macro_f1 = values["macro_f1"]
        if accuracy is None or macro_f1 is None:
            print(f"- {language}: test_size=0")
        else:
            print(
                f"- {language}: accuracy={accuracy:.4f}, "
                f"macro_f1={macro_f1:.4f}, test_size={values['test_size']}"
            )


if __name__ == "__main__":
    main()
