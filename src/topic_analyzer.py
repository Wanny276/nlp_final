"""Rule-based topic/aspect recognition."""

from __future__ import annotations

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
        "提高",
        "有用",
        "learn",
        "learned",
        "helpful",
        "useful",
        "skills",
        "understand",
        "understanding",
        "practical",
    },
}


def _keyword_position(text: str, keyword: str) -> int:
    position = text.lower().find(keyword.lower())
    return position if position >= 0 else len(text) + 1


def _evidence_snippet(text: str, keyword: str, window: int = 45) -> str:
    position = _keyword_position(text, keyword)
    if position > len(text):
        return text[: window * 2].strip()

    start = max(0, position - window)
    end = min(len(text), position + len(keyword) + window)
    return text[start:end].strip()


def detect_topic_evidence(text: str, max_topics: int | None = None) -> list[dict[str, object]]:
    """Detect course aspects with matched keywords and source evidence snippets."""

    normalized = text.lower()
    evidence_rows: list[dict[str, object]] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        matched = [keyword for keyword in keywords if keyword in normalized]
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
    """Detect topics by keyword hits."""

    return [str(item["aspect"]) for item in detect_topic_evidence(text, max_topics=max_topics)]
