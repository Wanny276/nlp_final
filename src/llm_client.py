"""LLM API client with a deterministic local fallback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - used only before dependencies are installed
    requests = None


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    timeout: int = 20

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            timeout=int(os.getenv("LLM_TIMEOUT", "20")),
        )


def build_single_review_prompt(analysis: dict[str, Any]) -> str:
    """Build a structured prompt for one review."""

    return f"""你是一个高校教学评价分析助手。请根据下面的学生课程评价，结合系统已经识别出的情感倾向、主题类别和关键词，生成简洁的分析结果。

学生评价：{analysis.get("text", "")}
语言类型：{analysis.get("language", "unknown")}
情感倾向：{analysis.get("sentiment", "")}
主题类别：{", ".join(analysis.get("topics", []))}
关键词：{", ".join(analysis.get("keywords", []))}

请输出 JSON 格式，包含 summary、problems、suggestions、risk_level。
要求：不要编造学生没有提到的问题；建议要具体；输出必须是合法 JSON。"""


def local_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """Generate a stable local summary when API is unavailable."""

    sentiment = analysis.get("sentiment", "neutral")
    language = analysis.get("language", "unknown")
    topics = analysis.get("topics", [])
    keywords = analysis.get("keywords", [])
    topic_text = "、".join(topics) if topics else "课程体验"
    keyword_text = "、".join(keywords[:5]) if keywords else "相关内容"

    if sentiment == "positive":
        summary = f"学生整体反馈较积极，主要认可{topic_text}方面。"
        risk_level = "low"
    elif sentiment == "negative":
        summary = f"学生反馈偏负面，问题集中在{topic_text}方面。"
        risk_level = "high"
    else:
        summary = f"学生反馈较为中性，涉及{topic_text}，同时包含肯定和改进建议。"
        risk_level = "middle"

    return {
        "summary": summary,
        "problems": [f"需要关注与“{keyword_text}”相关的反馈"],
        "suggestions": [f"结合学生原文，优先优化{topic_text}相关环节"],
        "risk_level": risk_level,
        "language": language,
        "source": "local_fallback",
    }


def call_llm_json(prompt: str, config: LLMConfig | None = None) -> dict[str, Any]:
    """Call an OpenAI-compatible chat completions API and parse JSON output."""

    config = config or LLMConfig.from_env()
    if not config.api_key or requests is None:
        raise RuntimeError("LLM API is not configured")

    url = config.base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=config.timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def generate_review_advice(analysis: dict[str, Any]) -> dict[str, Any]:
    """Generate LLM advice, falling back to local templates on failure."""

    prompt = build_single_review_prompt(analysis)
    try:
        result = call_llm_json(prompt)
        result.setdefault("source", "llm_api")
        return result
    except Exception:
        return local_summary(analysis)
