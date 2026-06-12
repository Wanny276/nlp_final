"""带确定性本地兜底的大模型 API 客户端。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 仅在依赖未安装时使用
    load_dotenv = None

try:
    import requests
except ImportError:  # pragma: no cover - 仅在依赖未安装时使用
    requests = None


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://chat.ecnu.edu.cn/open/api/v1"
    model: str = "ecnu-plus"
    timeout: int = 20

    @classmethod
    def from_env(cls) -> "LLMConfig":
        if load_dotenv is not None:
            dotenv_path = Path.cwd() / ".env"
            if not dotenv_path.exists():
                dotenv_path = Path(__file__).resolve().parents[1] / ".env"
            load_dotenv(dotenv_path=dotenv_path)

        return cls(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", "https://chat.ecnu.edu.cn/open/api/v1"),
            model=os.getenv("LLM_MODEL", "ecnu-plus"),
            timeout=int(os.getenv("LLM_TIMEOUT", "20")),
        )


def build_single_review_prompt(analysis: dict[str, Any]) -> str:
    """为单条评价构建结构化提示词。"""

    topic_evidence = json.dumps(analysis.get("topic_evidence", []), ensure_ascii=False)
    similar_reviews = json.dumps(analysis.get("similar_reviews", []), ensure_ascii=False)
    return f"""你是一个高校教学评价分析助手。请根据下面的学生课程评价，结合系统已经识别出的情感倾向、主题类别和关键词，生成简洁的分析结果。

学生评价：{analysis.get("text", "")}
语言类型：{analysis.get("language", "unknown")}
情感倾向：{analysis.get("sentiment", "")}
主题类别：{", ".join(analysis.get("topics", []))}
关键词：{", ".join(analysis.get("keywords", []))}
主题证据：{topic_evidence}
相似评论：{similar_reviews}

请输出合法 JSON 对象，包含 summary、problems、suggestions、risk_level。
problems 每项包含 aspect、description、evidence；suggestions 每项包含 aspect、suggestion、evidence。
要求：不要编造学生没有提到的问题；aspect 优先来自主题证据；evidence 必须引用学生评价、主题证据或相似评论中的原文片段；正面评价也要给出保持优势或继续优化的建议。"""


def local_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """在 API 不可用时生成稳定的本地总结。"""

    sentiment = analysis.get("sentiment", "neutral")
    language = analysis.get("language", "unknown")
    topics = analysis.get("topics", [])
    keywords = analysis.get("keywords", [])
    topic_text = "、".join(topics) if topics else "课程体验"
    keyword_text = "、".join(keywords[:5]) if keywords else "相关内容"
    aspect = str(topics[0]) if topics else "课程体验"
    evidence_items = analysis.get("topic_evidence", [])
    evidence = ""
    if evidence_items:
        evidence = str(evidence_items[0].get("evidence", ""))
    if not evidence:
        evidence = str(analysis.get("text", ""))[:80]

    if sentiment == "positive":
        summary = f"学生整体反馈较积极，主要认可{topic_text}方面。"
        risk_level = "low"
        problem_text = f"暂未发现明显问题，仍可继续关注与“{keyword_text}”相关的体验"
        suggestion_text = f"保持{topic_text}相关优势，并结合学生原文继续优化课程体验"
    elif sentiment == "negative":
        summary = f"学生反馈偏负面，问题集中在{topic_text}方面。"
        risk_level = "high"
        problem_text = f"需要关注与“{keyword_text}”相关的负面反馈"
        suggestion_text = f"优先优化{topic_text}相关环节，降低学生学习阻力"
    else:
        summary = f"学生反馈较为中性，涉及{topic_text}，同时包含肯定和改进建议。"
        risk_level = "middle"
        problem_text = f"需要进一步区分与“{keyword_text}”相关的肯定点和改进点"
        suggestion_text = f"保留{topic_text}中被认可的部分，同时处理学生提出的具体不便"

    return {
        "summary": summary,
        "problems": [
            {
                "aspect": aspect,
                "description": problem_text,
                "evidence": evidence,
            }
        ],
        "suggestions": [
            {
                "aspect": aspect,
                "suggestion": suggestion_text,
                "evidence": evidence,
            }
        ],
        "risk_level": risk_level,
        "language": language,
        "source": "local_fallback",
    }


def call_llm_json(prompt: str, config: LLMConfig | None = None) -> dict[str, Any]:
    """调用配置的大模型聊天接口，并解析 JSON 输出。"""

    config = config or LLMConfig.from_env()
    if not config.api_key or requests is None:
        raise RuntimeError("未配置大模型 API")

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
    """生成大模型建议，失败时回退到本地模板。"""

    prompt = build_single_review_prompt(analysis)
    try:
        result = call_llm_json(prompt)
        result.setdefault("source", "llm_api")
        return result
    except Exception:
        return local_summary(analysis)
