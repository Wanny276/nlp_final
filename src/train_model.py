"""Train a TF-IDF + Logistic Regression sentiment model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .preprocess import load_stopwords, preprocess_many


def train(data_path: str | Path, model_dir: str | Path = "models") -> dict[str, float]:
    """Train and save the sentiment classifier."""

    import joblib
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

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

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_vec, y_train)

    predictions = model.predict(x_test_vec)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "sentiment_model.pkl")
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_reviews.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    metrics = train(args.data, args.model_dir)
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

