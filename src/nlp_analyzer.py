"""Core NLP analysis orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .keyword_extractor import keywords_only
from .llm_client import generate_review_advice
from .preprocess import detect_language, load_stopwords, preprocess_text
from .similarity import find_similar_reviews
from .topic_analyzer import detect_topics


POSITIVE_HINTS = {"清楚", "有用", "帮助", "很好", "不错", "详细", "收获", "喜欢", "积极", "互动"}
NEGATIVE_HINTS = {"太多", "麻烦", "复杂", "报错", "不明确", "抓不到", "紧", "快", "困难", "压力"}
NEGATION_HINTS = {"但是", "但", "不过", "希望", "如果"}
ENGLISH_POSITIVE_HINTS = {
    "clear",
    "helpful",
    "useful",
    "excellent",
    "great",
    "good",
    "amazing",
    "practical",
    "engaging",
    "well organized",
    "recommend",
}
ENGLISH_NEGATIVE_HINTS = {
    "confusing",
    "unclear",
    "difficult",
    "hard",
    "boring",
    "outdated",
    "poor",
    "bad",
    "bug",
    "bugs",
    "too fast",
    "too many",
    "not clear",
    "lack",
}
ENGLISH_MIXED_HINTS = {"but", "however", "although", "wish", "could", "while"}
SENTIMENT_MODEL: Any | None = None
TFIDF_VECTORIZER: Any | None = None
MODEL_LOAD_ATTEMPTED = False
MIXED_MODEL_OVERRIDE_THRESHOLD = 0.78


def _sentiment_hint_counts(text: str) -> tuple[int, int]:
    normalized = text.lower()
    positive_hits = sum(1 for word in POSITIVE_HINTS if word in text)
    negative_hits = sum(1 for word in NEGATIVE_HINTS if word in text)
    positive_hits += sum(1 for word in ENGLISH_POSITIVE_HINTS if word in normalized)
    negative_hits += sum(1 for word in ENGLISH_NEGATIVE_HINTS if word in normalized)
    return positive_hits, negative_hits


def _has_mixed_signal(text: str) -> bool:
    normalized = text.lower()
    return any(word in text for word in NEGATION_HINTS) or any(
        word in normalized for word in ENGLISH_MIXED_HINTS
    )


def _is_balanced_mixed_review(text: str) -> bool:
    positive_hits, negative_hits = _sentiment_hint_counts(text)
    return (
        positive_hits > 0
        and negative_hits > 0
        and _has_mixed_signal(text)
        and negative_hits < positive_hits + 2
    )


def rule_based_sentiment(text: str) -> tuple[str, float]:
    """A lightweight sentiment fallback used before model training."""

    positive_hits, negative_hits = _sentiment_hint_counts(text)
    has_mixed_signal = _has_mixed_signal(text)

    if positive_hits > 0 and negative_hits > 0 and has_mixed_signal:
        if negative_hits >= positive_hits + 2:
            return "negative", min(0.95, 0.65 + negative_hits * 0.1)
        return "neutral", 0.72
    if positive_hits > negative_hits and not has_mixed_signal:
        return "positive", min(0.95, 0.65 + positive_hits * 0.1)
    if negative_hits > positive_hits:
        return "negative", min(0.95, 0.65 + negative_hits * 0.1)
    if positive_hits > negative_hits and has_mixed_signal:
        return "neutral", 0.65
    return "neutral", 0.6


def model_based_sentiment(processed_text: str, model_dir: str | Path = "models") -> tuple[str, float] | None:
    """Predict sentiment with a saved TF-IDF + Logistic Regression model."""

    global MODEL_LOAD_ATTEMPTED, SENTIMENT_MODEL, TFIDF_VECTORIZER

    if not MODEL_LOAD_ATTEMPTED:
        SENTIMENT_MODEL, TFIDF_VECTORIZER = load_model_if_available(model_dir)
        MODEL_LOAD_ATTEMPTED = True

    if SENTIMENT_MODEL is None or TFIDF_VECTORIZER is None or not processed_text:
        return None

    try:
        features = TFIDF_VECTORIZER.transform([processed_text])
        prediction = SENTIMENT_MODEL.predict(features)[0]
    except Exception:
        SENTIMENT_MODEL = None
        TFIDF_VECTORIZER = None
        return None

    confidence = 0.6
    if hasattr(SENTIMENT_MODEL, "predict_proba"):
        try:
            probabilities = SENTIMENT_MODEL.predict_proba(features)[0]
            confidence = float(max(probabilities))
        except Exception:
            confidence = 0.6

    return str(prediction), confidence


def predict_sentiment(text: str, processed_text: str) -> tuple[str, float, str]:
    """Use the trained model when available, otherwise fall back to rules."""

    rule_sentiment, rule_confidence = rule_based_sentiment(text)
    model_result = model_based_sentiment(processed_text)
    if model_result is not None:
        model_sentiment, model_confidence = model_result
        if (
            model_sentiment != "neutral"
            and _is_balanced_mixed_review(text)
            and model_confidence < MIXED_MODEL_OVERRIDE_THRESHOLD
        ):
            return "neutral", 0.72, "hybrid"
        if model_sentiment != rule_sentiment and model_confidence < 0.55:
            return rule_sentiment, max(rule_confidence, model_confidence), "hybrid"
        if (
            model_sentiment != rule_sentiment
            and rule_confidence >= 0.75
            and model_confidence < 0.68
        ):
            return rule_sentiment, rule_confidence, "hybrid"
        return model_sentiment, model_confidence, "model"

    return rule_sentiment, rule_confidence, "rule"


def analyze_review(text: str, reference_reviews: list[str] | None = None, use_llm: bool = True) -> dict[str, Any]:
    """Analyze one course review."""

    stopwords = load_stopwords()
    processed = preprocess_text(text, stopwords=stopwords)
    language = detect_language(text)
    sentiment, confidence, sentiment_source = predict_sentiment(text, processed)
    topics = detect_topics(text)
    keywords = keywords_only(text, top_k=6, stopwords=stopwords)
    similar_reviews = find_similar_reviews(text, reference_reviews or [], top_k=3)

    analysis: dict[str, Any] = {
        "text": text,
        "language": language,
        "processed_text": processed,
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "sentiment_source": sentiment_source,
        "topics": topics,
        "keywords": keywords,
        "similar_reviews": [
            {"text": review, "score": round(score, 3)}
            for review, score in similar_reviews
            if score > 0
        ],
    }

    if use_llm:
        analysis["llm_advice"] = generate_review_advice(analysis)

    return analysis


def analyze_batch(texts: list[str], use_llm: bool = False) -> list[dict[str, Any]]:
    """Analyze many reviews."""

    return [analyze_review(text, reference_reviews=texts, use_llm=use_llm) for text in texts]


def sentiment_distribution(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count sentiment labels."""

    distribution = {"positive": 0, "neutral": 0, "negative": 0}
    for result in results:
        label = result.get("sentiment", "neutral")
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def topic_distribution(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count detected topics."""

    distribution: dict[str, int] = {}
    for result in results:
        for topic in result.get("topics", []):
            distribution[topic] = distribution.get(topic, 0) + 1
    return dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))


def load_model_if_available(model_dir: str | Path = "models") -> tuple[Any | None, Any | None]:
    """Reserved hook for trained sklearn model loading."""

    try:
        import joblib

        model_path = Path(model_dir) / "sentiment_model.pkl"
        vectorizer_path = Path(model_dir) / "tfidf_vectorizer.pkl"
        if model_path.exists() and vectorizer_path.exists():
            return joblib.load(model_path), joblib.load(vectorizer_path)
    except Exception:
        return None, None
    return None, None
