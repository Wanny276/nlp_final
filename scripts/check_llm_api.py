"""Check whether the configured LLM API works with the project prompt."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_client import LLMConfig, generate_review_advice


def main() -> None:
    config = LLMConfig.from_env()
    if not config.api_key:
        raise RuntimeError("没有读取到 LLM_API_KEY，请检查 .env 是否在项目根目录")

    analysis = {
        "text": "老师讲课逻辑清晰，案例很实用，课堂互动也很多。",
        "language": "zh",
        "sentiment": "positive",
        "topics": ["授课方式", "教学内容"],
        "keywords": ["讲课", "逻辑", "案例", "互动"],
        "topic_evidence": [
            {
                "aspect": "授课方式",
                "keywords": ["讲课", "互动"],
                "evidence": "老师讲课逻辑清晰，课堂互动也很多。",
            },
            {
                "aspect": "教学内容",
                "keywords": ["案例"],
                "evidence": "案例很实用。",
            },
        ],
        "similar_reviews": [],
    }

    result = generate_review_advice(analysis, config=config)
    print("source=", result.get("source"))
    print("risk_level=", result.get("risk_level"))
    print("summary=", result.get("summary"))
    suggestions = result.get("suggestions") or []
    if suggestions:
        print("first_suggestion=", suggestions[0])

    if result.get("source") != "llm_api":
        raise RuntimeError("LLM API 未通过项目级检查，当前结果来自本地兜底")


if __name__ == "__main__":
    main()
