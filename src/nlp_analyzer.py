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


def rule_based_sentiment(text: str) -> tuple[str, float]:
    """A lightweight sentiment fallback used before model training."""

    normalized = text.lower()
    positive_hits = sum(1 for word in POSITIVE_HINTS if word in text)
    negative_hits = sum(1 for word in NEGATIVE_HINTS if word in text)
    positive_hits += sum(1 for word in ENGLISH_POSITIVE_HINTS if word in normalized)
    negative_hits += sum(1 for word in ENGLISH_NEGATIVE_HINTS if word in normalized)
    has_mixed_signal = any(word in text for word in NEGATION_HINTS) or any(
        word in normalized for word in ENGLISH_MIXED_HINTS
    )

    if positive_hits > negative_hits and not has_mixed_signal:
        return "positive", min(0.95, 0.65 + positive_hits * 0.1)
    if negative_hits > positive_hits:
        return "negative", min(0.95, 0.65 + negative_hits * 0.1)
    if positive_hits > negative_hits and has_mixed_signal:
        return "neutral", 0.65
    return "neutral", 0.6


def analyze_review(text: str, reference_reviews: list[str] | None = None, use_llm: bool = True) -> dict[str, Any]:
    """Analyze one course review."""

    stopwords = load_stopwords()
    processed = preprocess_text(text, stopwords=stopwords)
    language = detect_language(text)
    sentiment, confidence = rule_based_sentiment(text)
    topics = detect_topics(text)
    keywords = keywords_only(text, top_k=6, stopwords=stopwords)
    similar_reviews = find_similar_reviews(text, reference_reviews or [], top_k=3)

    analysis: dict[str, Any] = {
        "text": text,
        "language": language,
        "processed_text": processed,
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
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
