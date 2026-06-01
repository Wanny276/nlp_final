"""Train and compare TF-IDF based sentiment classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preprocess import load_stopwords, preprocess_many


def train(data_path: str | Path, model_dir: str | Path = "models") -> dict[str, float]:
    """Train several classifiers and save the best one."""

    import joblib
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Training data must contain 'text' and 'label' columns")

    stopwords = load_stopwords()
    texts = preprocess_many(df["text"].fillna("").tolist(), stopwords=stopwords)
    labels = df["label"].fillna("neutral").tolist()

    stratify = labels if len(set(labels)) > 1 and min(labels.count(label) for label in set(labels)) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
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
    }
    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_reviews.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    metrics = train(args.data, args.model_dir)
    print("model comparison:")
    for name, result in metrics["results"].items():
        print(f"- {name}: accuracy={result['accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}")
    print(f"best_model={metrics['best_model']}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
