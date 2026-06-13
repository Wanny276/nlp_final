"""核心 NLP 分析流程编排。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .bert_sentiment import BertConfig, bert_status, predict_bert
from .keyword_extractor import keywords_only
from .llm_client import generate_review_advice
from .preprocess import detect_language, load_stopwords, preprocess_text
from .similarity import find_similar_reviews
from .topic_analyzer import detect_topic_evidence


POSITIVE_HINTS = {"清楚", "有用", "帮助", "很好", "不错", "详细", "收获", "喜欢", "积极", "互动"}
NEGATIVE_HINTS = {
    "太多",
    "麻烦",
    "复杂",
    "报错",
    "不明确",
    "不太明确",
    "抓不到",
    "紧",
    "快",
    "困难",
    "压力",
}
NEGATION_HINTS = {"但是", "但", "不过", "希望", "如果"}
ENGLISH_POSITIVE_HINTS = {
    "clear",
    "clearly",
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
BERT_MIXED_OVERRIDE_THRESHOLD = 0.90
SUPPORTED_SENTIMENT_BACKENDS = {"auto", "bert", "tfidf", "rule"}

ENGLISH_NEGATION_PATTERN = (
    r"(?:not|never|isn't|isnt|wasn't|wasnt|aren't|arent|"
    r"is\s+not|was\s+not|are\s+not)"
)
ENGLISH_NEGATIVE_WORD_PATTERN = (
    r"(?:bad|poor|boring|difficult|hard|confusing|unclear|"
    r"useless|unhelpful|impossible)"
)
ENGLISH_POSITIVE_WORD_PATTERN = (
    r"(?:good|great|helpful|useful|clear|engaging|practical|"
    r"worthwhile|excellent|recommendable)"
)


def _sentiment_hint_counts(text: str) -> tuple[int, int]:
    normalized = text.lower()
    positive_hits = sum(1 for word in POSITIVE_HINTS if word in text)
    negative_hits = sum(1 for word in NEGATIVE_HINTS if word in text)
    positive_hits += sum(
        1 for word in ENGLISH_POSITIVE_HINTS if _contains_english_hint(normalized, word)
    )
    negative_hits += sum(
        1 for word in ENGLISH_NEGATIVE_HINTS if _contains_english_hint(normalized, word)
    )
    return positive_hits, negative_hits


def _contains_english_hint(normalized: str, hint: str) -> bool:
    words = [re.escape(word) for word in hint.lower().split()]
    pattern = r"\b" + r"\s+".join(words) + r"\b"
    return re.search(pattern, normalized) is not None


def _semantic_sentiment_override(text: str) -> tuple[str, float] | None:
    """Handle high-precision compositional cases the classifier often misses."""

    normalized = re.sub(r"\s+", " ", text.lower()).strip()

    # Negating a negative expression is usually a weak positive.
    if re.search(
        rf"\b{ENGLISH_NEGATION_PATTERN}\s+(?:very\s+|so\s+|that\s+)?"
        rf"{ENGLISH_NEGATIVE_WORD_PATTERN}\b",
        normalized,
    ):
        return "positive", 0.78
    if re.search(
        r"(?:不能说|并非|并不是|不是)(?:完全)?不"
        r"(?:好|清楚|有用|值得|实用|合理|简单|有帮助|有收获)",
        text,
    ) or re.search(
        r"(?:不能说|并非|并不是|不是)没有"
        r"(?:收获|帮助|价值|用处|学到东西)",
        text,
    ):
        return "positive", 0.78

    # Negating a positive expression is a direct negative signal.
    if re.search(
        rf"\b{ENGLISH_NEGATION_PATTERN}\s+(?:very\s+|so\s+|that\s+)?"
        rf"{ENGLISH_POSITIVE_WORD_PATTERN}\b",
        normalized,
    ):
        return "negative", 0.80
    if re.search(
        r"(?:不|没有)(?:太|很|那么)?"
        r"(?:好|清楚|有用|值得|实用|合理|简单|有帮助|有收获)",
        text,
    ):
        return "negative", 0.80

    english_sarcasm_marker = re.search(
        r"\b(?:great|wonderful|amazing|perfect|fantastic|"
        r"exactly what i needed|just what i needed)\b",
        normalized,
    )
    english_negative_context = re.search(
        r"\b(?:another\s+(?:impossible\s+)?(?:assignment|homework|deadline)|"
        r"more\s+(?:homework|assignments)|impossible|waste|broken|"
        r"nothing\s+(?:works|useful)|only\s+reads?\s+(?:the\s+)?slides?)\b",
        normalized,
    )
    chinese_sarcasm_marker = re.search(
        r"(?:真棒|太棒了|真精彩|太精彩了|真是太精彩了|可真)",
        text,
    )
    chinese_negative_context = re.search(
        r"(?:又多了|根本.*(?:做不完|听不懂)|只会|照着.*(?:PPT|课件)|"
        r"念.*(?:PPT|课件)|报错|浪费|没学到|太多|更糟)",
        text,
        re.IGNORECASE,
    )
    sarcastic_quotes = re.search(
        r"[“\"'](?:明确|清楚|简单|轻松|精彩|完美)[”\"']",
        text,
    )
    if (
        english_sarcasm_marker
        and english_negative_context
        or chinese_sarcasm_marker
        and chinese_negative_context
        or sarcastic_quotes
    ):
        return "negative", 0.84

    english_implicit_complaint = re.search(
        r"(?:\bonly\s+reads?\s+(?:the\s+)?slides?\b|"
        r"\ball\s+(?:the\s+)?(?:teacher|instructor)\s+does\s+is\s+read\b|"
        r"\bspen[dt]\s+more\s+time\s+.+\s+than\s+(?:learning|studying)\b|"
        r"\bmore\s+time\s+.+\s+than\s+(?:learning|studying)\b|"
        r"\bhad\s+to\s+learn\s+.+\s+on\s+my\s+own\b)",
        normalized,
    )
    chinese_implicit_complaint = re.search(
        r"(?:只会.*(?:念|读).*(?:PPT|课件)|"
        r"(?:上课)?基本都在(?:念|读).*(?:PPT|课件)|"
        r"(?:一节课)?大部分时间.*(?:处理|修|配置|排查).*(?:问题|环境|报错)|"
        r"全靠自己|几乎没有.*(?:讲解|反馈|例子))",
        text,
        re.IGNORECASE,
    )
    if english_implicit_complaint or chinese_implicit_complaint:
        return "negative", 0.82

    return None


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
    """模型训练前使用的轻量级情感兜底规则。"""

    semantic_override = _semantic_sentiment_override(text)
    if semantic_override is not None:
        return semantic_override

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
    """使用已保存的 TF-IDF + Logistic Regression 模型预测情感。"""

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


def model_based_sentiments(
    processed_texts: list[str],
    model_dir: str | Path = "models",
) -> list[tuple[str, float] | None] | None:
    """Run the TF-IDF fallback in one vectorized batch."""

    global MODEL_LOAD_ATTEMPTED, SENTIMENT_MODEL, TFIDF_VECTORIZER

    if not MODEL_LOAD_ATTEMPTED:
        SENTIMENT_MODEL, TFIDF_VECTORIZER = load_model_if_available(model_dir)
        MODEL_LOAD_ATTEMPTED = True
    if SENTIMENT_MODEL is None or TFIDF_VECTORIZER is None:
        return None

    valid_indexes = [index for index, text in enumerate(processed_texts) if text]
    results: list[tuple[str, float] | None] = [None] * len(processed_texts)
    if not valid_indexes:
        return results

    try:
        features = TFIDF_VECTORIZER.transform(
            [processed_texts[index] for index in valid_indexes]
        )
        predictions = SENTIMENT_MODEL.predict(features)
        probabilities = (
            SENTIMENT_MODEL.predict_proba(features)
            if hasattr(SENTIMENT_MODEL, "predict_proba")
            else None
        )
    except Exception:
        SENTIMENT_MODEL = None
        TFIDF_VECTORIZER = None
        return None

    for position, (index, prediction) in enumerate(zip(valid_indexes, predictions)):
        confidence = (
            float(max(probabilities[position])) if probabilities is not None else 0.6
        )
        results[index] = (str(prediction), confidence)
    return results


def bert_based_sentiments(texts: list[str]) -> list[tuple[str, float, str]] | None:
    predictions = predict_bert(texts)
    if predictions is None:
        return None
    return [
        (prediction.label, prediction.confidence, prediction.device)
        for prediction in predictions
    ]


def bert_based_sentiment(text: str) -> tuple[str, float, str] | None:
    predictions = bert_based_sentiments([text])
    return predictions[0] if predictions else None


def configured_sentiment_backend() -> str:
    BertConfig.from_env()
    backend = os.getenv("SENTIMENT_BACKEND", "auto").strip().lower()
    return backend if backend in SUPPORTED_SENTIMENT_BACKENDS else "auto"


def _apply_rule_correction(
    text: str,
    model_sentiment: str,
    model_confidence: float,
    source: str,
    device: str,
) -> tuple[str, float, str, str]:
    semantic_override = _semantic_sentiment_override(text)
    if semantic_override is not None:
        sentiment, confidence = semantic_override
        return sentiment, confidence, f"{source}+rule", device

    rule_sentiment, rule_confidence = rule_based_sentiment(text)
    corrected_source = f"{source}+rule"
    mixed_threshold = (
        BERT_MIXED_OVERRIDE_THRESHOLD
        if source == "bert"
        else MIXED_MODEL_OVERRIDE_THRESHOLD
    )
    if (
        model_sentiment != "neutral"
        and _is_balanced_mixed_review(text)
        and model_confidence < mixed_threshold
    ):
        return "neutral", max(0.72, rule_confidence), corrected_source, device
    if (
        model_sentiment == "neutral"
        and rule_sentiment == "negative"
        and not _has_mixed_signal(text)
        and model_confidence < 0.75
    ):
        return "negative", rule_confidence, corrected_source, device
    if model_sentiment != rule_sentiment and model_confidence < 0.55:
        return (
            rule_sentiment,
            max(rule_confidence, model_confidence),
            corrected_source,
            device,
        )
    if (
        model_sentiment != rule_sentiment
        and rule_confidence >= 0.75
        and model_confidence < 0.68
    ):
        return rule_sentiment, rule_confidence, corrected_source, device
    return model_sentiment, model_confidence, source, device


def predict_sentiment(
    text: str,
    processed_text: str,
) -> tuple[str, float, str, str]:
    """Prefer BERT, then TF-IDF, and always retain a rule fallback."""

    backend = configured_sentiment_backend()
    if backend in {"auto", "bert"}:
        bert_result = bert_based_sentiment(text)
        if bert_result is not None:
            label, confidence, device = bert_result
            return _apply_rule_correction(
                text, label, confidence, "bert", device
            )

    if backend in {"auto", "bert", "tfidf"}:
        model_result = model_based_sentiment(processed_text)
        if model_result is not None:
            label, confidence = model_result
            return _apply_rule_correction(
                text, label, confidence, "tfidf", "cpu"
            )

    label, confidence = rule_based_sentiment(text)
    return label, confidence, "rule", "cpu"


def predict_sentiments(
    texts: list[str],
    processed_texts: list[str],
) -> list[tuple[str, float, str, str]]:
    """Select a backend once and classify a full batch."""

    if len(texts) != len(processed_texts):
        raise ValueError("texts and processed_texts must have the same length")
    if not texts:
        return []

    backend = configured_sentiment_backend()
    if backend in {"auto", "bert"}:
        bert_results = bert_based_sentiments(texts)
        if bert_results is not None and len(bert_results) == len(texts):
            return [
                _apply_rule_correction(text, label, confidence, "bert", device)
                for text, (label, confidence, device) in zip(texts, bert_results)
            ]

    if backend in {"auto", "bert", "tfidf"}:
        model_results = model_based_sentiments(processed_texts)
        if model_results is not None:
            predictions: list[tuple[str, float, str, str]] = []
            for text, model_result in zip(texts, model_results):
                if model_result is None:
                    label, confidence = rule_based_sentiment(text)
                    predictions.append((label, confidence, "rule", "cpu"))
                    continue
                label, confidence = model_result
                predictions.append(
                    _apply_rule_correction(
                        text, label, confidence, "tfidf", "cpu"
                    )
                )
            return predictions

    return [
        (*rule_based_sentiment(text), "rule", "cpu")
        for text in texts
    ]


def _build_analysis(
    text: str,
    processed: str,
    prediction: tuple[str, float, str, str],
    stopwords: set[str],
    reference_reviews: list[str],
    use_llm: bool,
) -> dict[str, Any]:
    sentiment, confidence, sentiment_source, sentiment_device = prediction
    language = detect_language(text)
    topic_evidence = detect_topic_evidence(text)
    topics = [str(item["aspect"]) for item in topic_evidence]
    keywords = keywords_only(text, top_k=6, stopwords=stopwords)
    similar_reviews = find_similar_reviews(text, reference_reviews, top_k=3)

    analysis: dict[str, Any] = {
        "text": text,
        "language": language,
        "processed_text": processed,
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "sentiment_source": sentiment_source,
        "sentiment_device": sentiment_device,
        "topics": topics,
        "topic_evidence": topic_evidence,
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


def analyze_review(
    text: str,
    reference_reviews: list[str] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Analyze one course review."""

    stopwords = load_stopwords()
    processed = preprocess_text(text, stopwords=stopwords)
    prediction = predict_sentiment(text, processed)
    return _build_analysis(
        text,
        processed,
        prediction,
        stopwords,
        reference_reviews or [],
        use_llm,
    )


def analyze_batch(
    texts: list[str],
    use_llm: bool = False,
    reference_reviews: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Analyze reviews with one batched sentiment inference call."""

    stopwords = load_stopwords()
    processed_texts = [
        preprocess_text(text, stopwords=stopwords)
        for text in texts
    ]
    predictions = predict_sentiments(texts, processed_texts)
    references = texts if reference_reviews is None else reference_reviews
    return [
        _build_analysis(
            text,
            processed,
            prediction,
            stopwords,
            references,
            use_llm,
        )
        for text, processed, prediction in zip(texts, processed_texts, predictions)
    ]


def sentiment_runtime_status() -> dict[str, Any]:
    """Return lightweight backend availability information for the UI."""

    backend = configured_sentiment_backend()
    bert = bert_status()
    tfidf_available = (
        Path("models/sentiment_model.pkl").exists()
        and Path("models/tfidf_vectorizer.pkl").exists()
    )
    if backend == "rule":
        active = "rule"
    elif backend == "tfidf":
        active = "tfidf" if tfidf_available else "rule"
    elif bert["ready"]:
        active = "bert"
    elif tfidf_available:
        active = "tfidf"
    else:
        active = "rule"
    return {
        "configured_backend": backend,
        "active_backend": active,
        "bert": bert,
        "tfidf_available": tfidf_available,
    }


def sentiment_distribution(results: list[dict[str, Any]]) -> dict[str, int]:
    """统计情感标签数量。"""

    distribution = {"positive": 0, "neutral": 0, "negative": 0}
    for result in results:
        label = result.get("sentiment", "neutral")
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def topic_distribution(results: list[dict[str, Any]]) -> dict[str, int]:
    """统计识别出的主题数量。"""

    distribution: dict[str, int] = {}
    for result in results:
        for topic in result.get("topics", []):
            distribution[topic] = distribution.get(topic, 0) + 1
    return dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))


def load_model_if_available(model_dir: str | Path = "models") -> tuple[Any | None, Any | None]:
    """加载已训练 sklearn 模型的保留入口。"""

    try:
        import joblib

        model_path = Path(model_dir) / "sentiment_model.pkl"
        vectorizer_path = Path(model_dir) / "tfidf_vectorizer.pkl"
        if model_path.exists() and vectorizer_path.exists():
            return joblib.load(model_path), joblib.load(vectorizer_path)
    except Exception:
        return None, None
    return None, None
