"""Text cleaning and tokenization utilities."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import jieba
except ImportError:  # pragma: no cover - used only before dependencies are installed
    jieba = None


DEFAULT_STOPWORDS_PATH = Path("data/stopwords.txt")
NEGATION_WORDS = {"不", "没有", "不是", "没", "不要", "不能"}
ENGLISH_NEGATION_WORDS = {"no", "not", "never", "none", "without"}
ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "however",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "we",
    "were",
    "while",
    "with",
    "you",
    "your",
    "although",
    "could",
    "course",
    "teacher",
    "student",
    "students",
}
CHINESE_DOMAIN_TERMS = (
    "知识点",
    "不明确",
    "实验环境",
    "教学内容",
    "课程内容",
    "作业",
    "提交",
    "时间",
    "太多",
    "实验",
    "环境",
    "配置",
    "复杂",
    "步骤",
    "麻烦",
    "内容",
    "难度",
    "重点",
    "案例",
    "复习",
    "概念",
    "讲课",
    "讲解",
    "语速",
    "互动",
    "解释",
    "清楚",
    "课堂",
    "氛围",
    "任务",
    "练习",
    "批改",
    "报告",
    "考试",
    "范围",
    "题型",
    "成绩",
    "代码",
    "运行",
    "报错",
    "示例",
    "实践",
    "收获",
    "帮助",
    "能力",
    "理解",
    "提高",
    "有用",
    "压力",
    "困难",
    "紧",
)


def load_stopwords(path: str | Path = DEFAULT_STOPWORDS_PATH) -> set[str]:
    """Load stopwords from a text file."""

    stopwords_path = Path(path)
    if not stopwords_path.exists():
        return set()

    return {
        line.strip()
        for line in stopwords_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def clean_text(text: object) -> str:
    """Normalize one raw review string."""

    if text is None:
        return ""

    value = str(text).strip()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"\b[A-Za-z]_[A-Za-z]\b", " ", value)
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9，。！？；、,.!?;:：\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def detect_language(text: object) -> str:
    """Detect whether a review is Chinese, English, or mixed."""

    cleaned = clean_text(text)
    chinese_chars = len(re.findall(r"[\u4e00-\u9fa5]", cleaned))
    english_chars = len(re.findall(r"[A-Za-z]", cleaned))
    if chinese_chars and english_chars:
        return "mixed"
    if chinese_chars:
        return "zh"
    if english_chars:
        return "en"
    return "unknown"


def _english_tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text)]


def _expand_chinese_token(token: str) -> list[str]:
    if not re.search(r"[\u4e00-\u9fa5]", token):
        return [token]

    matches = [term for term in CHINESE_DOMAIN_TERMS if term in token and term != token]
    if not matches:
        return [token]

    return matches


def tokenize(text: object, stopwords: set[str] | None = None) -> list[str]:
    """Clean, segment, and filter a Chinese, English, or mixed review."""

    stopwords = stopwords or set()
    cleaned = clean_text(text)
    if not cleaned:
        return []

    language = detect_language(cleaned)
    if language == "en":
        raw_tokens = _english_tokens(cleaned)
    elif jieba is not None:
        raw_tokens = jieba.cut_for_search(cleaned)
    else:
        raw_tokens = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", cleaned)

    merged_stopwords = stopwords | ENGLISH_STOPWORDS
    tokens: list[str] = []
    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", token):
            token = token.lower()
        if re.fullmatch(r"[，。！？；、,.!?;:：]+", token):
            continue
        expanded_tokens = _expand_chinese_token(token)
        for expanded_token in expanded_tokens:
            if expanded_token in merged_stopwords and expanded_token not in NEGATION_WORDS | ENGLISH_NEGATION_WORDS:
                continue
            tokens.append(expanded_token)

    return tokens


def preprocess_text(text: object, stopwords: set[str] | None = None) -> str:
    """Return whitespace-joined tokens for vectorizers."""

    return " ".join(tokenize(text, stopwords))


def preprocess_many(texts: list[object], stopwords: set[str] | None = None) -> list[str]:
    """Preprocess a list of texts."""

    return [preprocess_text(text, stopwords) for text in texts]
