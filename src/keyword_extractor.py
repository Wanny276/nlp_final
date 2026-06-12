"""关键词提取辅助函数。"""

from __future__ import annotations

from collections import Counter

from .preprocess import tokenize


def extract_keywords(texts: list[str] | str, top_k: int = 10, stopwords: set[str] | None = None) -> list[tuple[str, int]]:
    """从单条文本或文本列表中提取高频词。"""

    if isinstance(texts, str):
        corpus = [texts]
    else:
        corpus = texts

    counter: Counter[str] = Counter()
    for text in corpus:
        counter.update(tokenize(text, stopwords=stopwords))

    return counter.most_common(top_k)


def keywords_only(texts: list[str] | str, top_k: int = 10, stopwords: set[str] | None = None) -> list[str]:
    """只返回关键词字符串。"""

    return [word for word, _ in extract_keywords(texts, top_k=top_k, stopwords=stopwords)]
