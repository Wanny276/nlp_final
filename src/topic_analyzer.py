"""Rule-based topic/aspect recognition."""

from __future__ import annotations

TOPIC_KEYWORDS: dict[str, set[str]] = {
    "教学内容": {
        "内容",
        "知识点",
        "难度",
        "重点",
        "案例",
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


def detect_topics(text: str, max_topics: int | None = None) -> list[str]:
    """Detect topics by keyword hits."""

    normalized = text.lower()
    scores: list[tuple[str, int]] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in normalized)
        if score > 0:
            scores.append((topic, score))

    scores.sort(key=lambda item: (-item[1], item[0]))
    topics = [topic for topic, _ in scores]
    if max_topics is not None:
        return topics[:max_topics]
    return topics
