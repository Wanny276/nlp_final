"""基于规则的主题和课程维度识别。"""

from __future__ import annotations

import re


TOPIC_KEYWORDS: dict[str, set[str]] = {
    "教学内容": {
        "内容",
        "知识点",
        "难度",
        "重点",
        "案例",
        "示例",
        "ppt",
        "复习",
        "概念",
        "content",
        "concept",
        "concepts",
        "material",
        "materials",
        "lecture",
        "lectures",
        "syllabus",
        "difficult",
        "difficulty",
    },
    "授课方式": {
        "讲课",
        "讲解",
        "语速",
        "互动",
        "解释",
        "清楚",
        "课堂",
        "氛围",
        "instructor",
        "teaching",
        "explain",
        "explanation",
        "clear",
        "interactive",
        "pace",
        "video",
        "videos",
    },
    "作业任务": {
        "作业",
        "任务",
        "练习",
        "提交",
        "批改",
        "报告",
        "assignment",
        "assignments",
        "homework",
        "exercise",
        "exercises",
        "deadline",
        "deadlines",
        "submission",
        "peer review",
    },
    "考试安排": {
        "考试",
        "范围",
        "题型",
        "复习",
        "成绩",
        "重点",
        "quiz",
        "quizzes",
        "exam",
        "exams",
        "test",
        "tests",
        "grade",
        "grading",
        "certificate",
    },
    "实验实践": {
        "实验",
        "环境",
        "代码",
        "配置",
        "运行",
        "报错",
        "示例",
        "实践",
        "lab",
        "labs",
        "project",
        "projects",
        "code",
        "coding",
        "programming",
        "setup",
        "environment",
        "bug",
        "bugs",
    },
    "学习收获": {
        "收获",
        "帮助",
        "能力",
        "理解",
        "掌握",
        "学会",
        "提高",
        "有用",
        "learn",
        "learned",
        "master",
        "mastered",
        "helpful",
        "useful",
        "skills",
        "understand",
        "understanding",
        "practical",
    },
}

TEACHING_METHOD_CONTEXT = {
    "老师",
    "教师",
    "讲",
    "解释",
    "课堂",
    "instructor",
    "teacher",
    "teaching",
    "explain",
    "explanation",
    "lecture",
    "lectures",
    "video",
    "videos",
    "pace",
    "interactive",
}
EXAM_CONTEXT = {
    "考试",
    "范围",
    "题型",
    "复习",
    "成绩",
    "quiz",
    "quizzes",
    "exam",
    "exams",
    "test",
    "tests",
    "grade",
    "grading",
}
PRACTICE_CONTEXT = {
    "实验",
    "环境",
    "代码",
    "配置",
    "运行",
    "报错",
    "实践",
    "lab",
    "labs",
    "project",
    "projects",
    "code",
    "coding",
    "programming",
    "setup",
    "environment",
    "bug",
    "bugs",
}


def _keyword_pattern(keyword: str) -> re.Pattern[str] | None:
    if not re.search(r"[A-Za-z]", keyword):
        return None
    words = [re.escape(word) for word in keyword.split()]
    return re.compile(
        r"(?<![A-Za-z0-9])" + r"\s+".join(words) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def _keyword_matches(text: str, keyword: str) -> bool:
    pattern = _keyword_pattern(keyword)
    if pattern is not None:
        return pattern.search(text) is not None
    return keyword in text


def _contextual_topic_matches(topic: str, matched: list[str], text: str) -> list[str]:
    matched_set = set(matched)
    if topic == "授课方式" and matched_set.issubset({"clear", "清楚"}):
        if not any(_keyword_matches(text, anchor) for anchor in TEACHING_METHOD_CONTEXT):
            return []
    if topic == "考试安排" and matched_set == {"重点"}:
        if not any(_keyword_matches(text, anchor) for anchor in EXAM_CONTEXT):
            return []
    if topic == "实验实践" and matched_set == {"示例"}:
        if not any(_keyword_matches(text, anchor) for anchor in PRACTICE_CONTEXT):
            return []
    return matched


def _keyword_position(text: str, keyword: str) -> int:
    pattern = _keyword_pattern(keyword)
    if pattern is not None:
        match = pattern.search(text)
        return match.start() if match is not None else len(text)
    position = text.find(keyword)
    return position if position >= 0 else len(text)


def _evidence_snippet(text: str, keyword: str, window: int = 45) -> str:
    position = _keyword_position(text, keyword)
    if position >= len(text):
        return text[: window * 2].strip()

    start = max(0, position - window)
    end = min(len(text), position + len(keyword) + window)
    while start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        start -= 1
    while end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end += 1
    return text[start:end].strip()


def detect_topic_evidence(text: str, max_topics: int | None = None) -> list[dict[str, object]]:
    """根据命中关键词和原文证据片段识别课程维度。"""

    evidence_rows: list[dict[str, object]] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        matched = [keyword for keyword in keywords if _keyword_matches(text, keyword)]
        matched = _contextual_topic_matches(topic, matched, text)
        if not matched:
            continue

        matched.sort(key=lambda keyword: (_keyword_position(text, keyword), keyword))
        evidence_rows.append(
            {
                "aspect": topic,
                "keywords": matched,
                "evidence": _evidence_snippet(text, matched[0]),
                "score": len(matched),
            }
        )

    evidence_rows.sort(key=lambda item: (-int(item["score"]), str(item["aspect"])))
    if max_topics is not None:
        evidence_rows = evidence_rows[:max_topics]

    return [
        {
            "aspect": item["aspect"],
            "keywords": item["keywords"],
            "evidence": item["evidence"],
        }
        for item in evidence_rows
    ]


def detect_topics(text: str, max_topics: int | None = None) -> list[str]:
    """根据关键词命中结果识别主题。"""

    return [str(item["aspect"]) for item in detect_topic_evidence(text, max_topics=max_topics)]
