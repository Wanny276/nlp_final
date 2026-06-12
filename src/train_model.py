"""训练并比较基于 TF-IDF 的情感分类器。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli_utils import ChineseArgumentParser
from .preprocess import detect_language, load_stopwords, preprocess_many


LABEL_ORDER = ["negative", "neutral", "positive"]
VECTORIZER_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 30000,
    "sublinear_tf": True,
}


def _language_metrics(
    y_true: list[str],
    y_pred: list[str],
    languages: list[str],
) -> dict[str, dict[str, float | int]]:
    from sklearn.metrics import accuracy_score, f1_score

    metrics: dict[str, dict[str, float | int]] = {}
    for language in sorted(set(languages)):
        indexes = [index for index, value in enumerate(languages) if value == language]
        if not indexes:
            continue

        true_values = [y_true[index] for index in indexes]
        pred_values = [y_pred[index] for index in indexes]
        metrics[language] = {
            "accuracy": float(accuracy_score(true_values, pred_values)),
            "macro_f1": float(
                f1_score(true_values, pred_values, average="macro", zero_division=0)
            ),
            "support": len(indexes),
        }

    return metrics


def train(data_path: str | Path, model_dir: str | Path = "models") -> dict[str, object]:
    """训练多个分类器并保存最优模型。"""

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
        raise ValueError("训练数据必须包含 'text' 和 'label' 列")

    stopwords = load_stopwords()
    texts = preprocess_many(df["text"].fillna("").tolist(), stopwords=stopwords)
    labels = df["label"].fillna("neutral").tolist()
    languages = (
        df["language"].fillna("unknown").astype(str).tolist()
        if "language" in df.columns
        else [detect_language(text) for text in df["text"].fillna("").tolist()]
    )

    stratify = labels if len(set(labels)) > 1 and min(labels.count(label) for label in set(labels)) > 1 else None
    x_train, x_test, y_train, y_test, _language_train, language_test = train_test_split(
        texts,
        labels,
        languages,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG)
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
    predictions_by_model: dict[str, list[str]] = {}
    for name, model in candidates.items():
        model.fit(x_train_vec, y_train)
        predictions = model.predict(x_test_vec)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        }
        trained_models[name] = model
        predictions_by_model[name] = [str(prediction) for prediction in predictions]

    best_name = max(results, key=lambda name: (results[name]["macro_f1"], results[name]["accuracy"]))
    best_metrics = results[best_name]
    best_predictions = predictions_by_model[best_name]

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models[best_name], output_dir / "sentiment_model.pkl")
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")

    metrics = {
        "best_model": best_name,
        "accuracy": best_metrics["accuracy"],
        "macro_f1": best_metrics["macro_f1"],
        "results": results,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "data_path": str(data_path),
        "vectorizer": {
            "ngram_range": list(VECTORIZER_CONFIG["ngram_range"]),
            "min_df": VECTORIZER_CONFIG["min_df"],
            "max_df": VECTORIZER_CONFIG["max_df"],
            "max_features": VECTORIZER_CONFIG["max_features"],
            "sublinear_tf": VECTORIZER_CONFIG["sublinear_tf"],
        },
        "labels": LABEL_ORDER,
        "confusion_matrix": confusion_matrix(
            y_test,
            best_predictions,
            labels=LABEL_ORDER,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            best_predictions,
            labels=LABEL_ORDER,
            zero_division=0,
            output_dict=True,
        ),
        "language_metrics": _language_metrics(y_test, best_predictions, language_test),
    }
    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = ChineseArgumentParser(description="训练并比较传统情感分类模型")
    parser.add_argument("--data", default="data/sample_reviews.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    metrics = train(args.data, args.model_dir)
    print("模型对比：")
    for name, result in metrics["results"].items():
        print(f"- {name}: 准确率={result['accuracy']:.4f}，宏平均F1={result['macro_f1']:.4f}")
    print(f"最佳模型={metrics['best_model']}")
    print(f"准确率={metrics['accuracy']:.4f}")
    print(f"宏平均F1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
