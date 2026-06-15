"""关键词提取辅助函数。"""

from __future__ import annotations

from collections import Counter

from .preprocess import tokenize


KEYWORD_STOPWORDS = {
    "不",
    "不是",
    "不能",
    "没有",
    "没",
    "至少",
    "基本",
    "课",
    "说",
    "让",
    "我",
    "我们",
    "no",
    "not",
    "never",
    "too",
    "many",
    "enough",
    "very",
    "extremely",
    "would",
}
ENGLISH_SINGULAR_EXCEPTIONS = {
    "analysis",
    "business",
    "class",
    "crs",
    "economics",
    "ethics",
    "graphics",
    "linguistics",
    "mathematics",
    "news",
    "physics",
    "politics",
    "process",
    "series",
    "statistics",
    "status",
    "stress",
}
ENGLISH_IRREGULAR_PLURALS = {
    "analyses": "analysis",
    "classes": "class",
    "processes": "process",
    "quizzes": "quiz",
    "statuses": "status",
}


def normalize_english_keyword(token: str) -> str:
    """Conservatively merge common English plural keyword variants."""

    normalized = token.lower()
    if not normalized.isascii() or not normalized.isalpha():
        return token
    if normalized in ENGLISH_SINGULAR_EXCEPTIONS:
        return normalized
    if normalized in ENGLISH_IRREGULAR_PLURALS:
        return ENGLISH_IRREGULAR_PLURALS[normalized]
    if len(normalized) > 4 and normalized.endswith("ies"):
        return normalized[:-3] + "y"
    if len(normalized) > 4 and normalized.endswith(("sses", "shes", "ches", "xes", "zes")):
        return normalized[:-2]
    if (
        len(normalized) > 3
        and normalized.endswith("s")
        and not normalized.endswith(("ss", "us", "is"))
    ):
        return normalized[:-1]
    return normalized


def extract_keywords(
    texts: list[str] | str,
    top_k: int = 10,
    stopwords: set[str] | None = None,
    normalize_english_plurals: bool = False,
) -> list[tuple[str, int]]:
    """从单条文本或文本列表中提取高频词。"""

    if isinstance(texts, str):
        corpus = [texts]
    else:
        corpus = texts

    counter: Counter[str] = Counter()
    for text in corpus:
        counter.update(
            normalize_english_keyword(token) if normalize_english_plurals else token
            for token in tokenize(text, stopwords=stopwords)
            if token.lower() not in KEYWORD_STOPWORDS
        )

    return counter.most_common(top_k)


def keywords_only(
    texts: list[str] | str,
    top_k: int = 10,
    stopwords: set[str] | None = None,
    normalize_english_plurals: bool = False,
) -> list[str]:
    """只返回关键词字符串。"""

    return [
        word
        for word, _ in extract_keywords(
            texts,
            top_k=top_k,
            stopwords=stopwords,
            normalize_english_plurals=normalize_english_plurals,
        )
    ]
