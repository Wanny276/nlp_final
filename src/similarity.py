"""相似评价检索。"""

from __future__ import annotations

import math
from collections import Counter

from .preprocess import tokenize


def _term_vector(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_similarity(text_a: str, text_b: str) -> float:
    """使用简单词频向量计算余弦相似度。"""

    vec_a = _term_vector(text_a)
    vec_b = _term_vector(text_b)
    if not vec_a or not vec_b:
        return 0.0

    shared = set(vec_a) & set(vec_b)
    numerator = sum(vec_a[token] * vec_b[token] for token in shared)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


def find_similar_reviews(query: str, reviews: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """按余弦相似度返回前 top-k 条相似评价。"""

    scored = [(review, cosine_similarity(query, review)) for review in reviews if review != query]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
